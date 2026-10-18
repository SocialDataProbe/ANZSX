import asyncio
import secrets
import threading
import time

import anthropic
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
)

from backend import database
from backend.batch_table import BatchTable, RunConfig
from backend.classifier import _parse_json_response, build_request_params, extract_text
from backend.cost import estimate_single_cost
from backend.rate_limiter import RateLimiter, backoff_seconds, start_tier_limits
from backend.templates import build_json_schema, build_system_prompt
from backend.verification import JEV_TOKEN_BUDGET, verify_extraction

MAX_RETRIES = 3
CIRCUIT_BREAKER = 10
FETCH_CHUNK = 50
FETCH_RETRIES = 5
DEFAULT_OUTPUT_ESTIMATE = 2_000
IDLE_PURGE_SECONDS = 6 * 3600

RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}
FATAL_STATUSES = {401, 403, 404}


class FatalRunError(Exception):
    pass


class BatchRun:
    def __init__(self, table: BatchTable, config: RunConfig):
        self.table = table
        self.config = config
        self.phase = "idle"  # idle | classifying | verifying | finished | stopped | failed
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_error = ""
        self.stop_reason = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.phase_total = 0
        self.phase_done = 0
        self.phase_errors = 0
        self.limiter: RateLimiter | None = None
        self.limits = start_tier_limits(config.model)
        self.last_activity = time.time()
        self.downloaded_version = -1

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def in_flight(self) -> int:
        return self.limiter.in_flight if self.limiter and self.running else 0

    def progress(self) -> dict:
        elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0.0
        rate = self.phase_done / elapsed * 60 if elapsed > 0 else 0.0
        remaining = self.phase_total - self.phase_done
        eta_min = remaining / rate if rate > 0 and self.running else None
        return {
            "phase": self.phase,
            "total": self.phase_total,
            "done": self.phase_done,
            "errors": self.phase_errors,
            "elapsed_s": elapsed,
            "docs_per_min": rate,
            "eta_min": eta_min,
            "in_flight": self.in_flight,
            "last_error": self.last_error,
            "stop_reason": self.stop_reason,
        }

    def _begin(self, phase: str, total: int):
        self.phase = phase
        self.phase_total = total
        self.phase_done = 0
        self.phase_errors = 0
        self.started_at = time.time()
        self.finished_at = None
        self.last_error = ""
        self.stop_reason = ""

    def _end(self, phase: str, reason: str = ""):
        self.phase = phase
        self.stop_reason = reason
        self.finished_at = time.time()
        self.last_activity = time.time()


# ── Run registry (server memory, keyed by the ?run= URL token) ──

_REGISTRY: dict[str, BatchRun] = {}
_REGISTRY_LOCK = threading.Lock()


def register_run(run: BatchRun) -> str:
    purge_idle_runs()
    token = secrets.token_urlsafe(9)
    with _REGISTRY_LOCK:
        _REGISTRY[token] = run
    return token


def get_run(token: str | None) -> BatchRun | None:
    if not token:
        return None
    with _REGISTRY_LOCK:
        run = _REGISTRY.get(token)
    if run:
        run.last_activity = time.time()
    return run


def remove_run(token: str):
    with _REGISTRY_LOCK:
        run = _REGISTRY.get(token)
        if run and not run.running:
            del _REGISTRY[token]


def purge_idle_runs():
    cutoff = time.time() - IDLE_PURGE_SECONDS
    with _REGISTRY_LOCK:
        for token in [t for t, r in _REGISTRY.items() if not r.running and r.last_activity < cutoff]:
            del _REGISTRY[token]


# ── Rate limit probe ──

def probe_rate_limits(api_key: str, model: str) -> dict:
    """One tiny request to read this key's real limits for the model."""
    client = anthropic.Anthropic(api_key=api_key, max_retries=1)
    try:
        raw = client.messages.with_raw_response.create(
            model=model, max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
        h = raw.headers
        limits = {
            "rpm": int(h.get("anthropic-ratelimit-requests-limit")),
            "itpm": int(h.get("anthropic-ratelimit-input-tokens-limit")),
            "otpm": int(h.get("anthropic-ratelimit-output-tokens-limit")),
            "source": "your API key",
        }
        return limits
    except Exception:
        return start_tier_limits(model)


# ── Starting runs ──

def start_classification(
    run: BatchRun,
    *,
    api_key: str,
    concurrency: int,
    output_estimate: float | None = None,
    jev_api_key: str | None = None,
) -> bool:
    if run.running:
        return False
    run.cancel.clear()

    def _target():
        try:
            completed = asyncio.run(_classify(run, api_key, concurrency, output_estimate))
        except FatalRunError as e:
            run._end("failed", str(e))
            return
        except Exception as e:
            run._end("failed", f"Unexpected error: {type(e).__name__}: {e}")
            return
        if completed and jev_api_key and run.table.rowids_for_jev():
            _run_verification(run, jev_api_key, api_key)

    run.thread = threading.Thread(target=_target, daemon=True)
    run.thread.start()
    return True


def _run_verification(run: BatchRun, jev_api_key: str, api_key: str | None):
    try:
        asyncio.run(_verify(run, jev_api_key, api_key))
    except FatalRunError as e:
        run._end("failed", str(e))
    except Exception as e:
        run._end("failed", f"Unexpected error: {type(e).__name__}: {e}")


# ── Shared helpers ──

async def _fetch_records(chunk: list[int]) -> dict[int, dict]:
    for attempt in range(FETCH_RETRIES + 1):
        try:
            return await asyncio.to_thread(database.get_records_by_rowids, chunk)
        except Exception as e:
            if attempt == FETCH_RETRIES:
                raise FatalRunError(
                    f"Couldn't read from the database after {FETCH_RETRIES} retries. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e
            await asyncio.sleep(backoff_seconds(attempt))


async def _produce(rowids: list[int], queue: asyncio.Queue, run: BatchRun, n_workers: int):
    try:
        for i in range(0, len(rowids), FETCH_CHUNK):
            if run.cancel.is_set():
                break
            chunk = rowids[i:i + FETCH_CHUNK]
            records = await _fetch_records(chunk)
            for rid in chunk:
                await queue.put((rid, records.get(rid)))
    finally:
        # Always release the workers, or a failed fetch would leave the run stuck.
        for _ in range(n_workers):
            await queue.put(None)


def _integrity_error(row: dict, record: dict | None) -> str:
    if record is None:
        return "Row not found in database (rowid changed?)"
    if (
        str(record["company_name"]) != str(row["company_name"])
        or str(record["year"]) != str(row["year"])
        or str(record["section_name"]) != str(row["section_name"])
    ):
        return (
            "Database row doesn't match CSV (company/year/section differ); "
            "the database may have been rebuilt"
        )
    return ""


def _retry_after(e: anthropic.APIStatusError) -> float | None:
    value = e.response.headers.get("retry-after") if e.response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


# ── Phase 1: classification ──

async def _classify(run: BatchRun, api_key: str, concurrency: int, output_estimate: float | None) -> bool:
    cfg = run.config
    rowids = run.table.rowids_to_process()
    run._begin("classifying", len(rowids))
    if not rowids:
        run._end("finished")
        return True

    client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=600)
    limiter = RateLimiter.for_model(cfg.model, max_concurrency=concurrency)
    if run.limits.get("source") == "your API key":
        limiter.update_from_headers({
            "anthropic-ratelimit-requests-limit": str(run.limits["rpm"]),
            "anthropic-ratelimit-input-tokens-limit": str(run.limits["itpm"]),
            "anthropic-ratelimit-output-tokens-limit": str(run.limits["otpm"]),
        })
    run.limiter = limiter

    system_prompt = build_system_prompt()
    schema = build_json_schema(list(cfg.taxonomy)) if cfg.taxonomy else None
    stats = run.table.usage_stats()
    state = {
        "token_ratio": 1.3,  # actual input tokens per (chars / 4); newer tokenizers run higher
        "output_est": (stats or {}).get("avg_output") or output_estimate or DEFAULT_OUTPUT_ESTIMATE,
        "consecutive_failures": 0,
        "fatal": None,
    }

    queue: asyncio.Queue = asyncio.Queue(maxsize=max(2 * concurrency, FETCH_CHUNK))

    def _row_failed(rowid: int, message: str, meta: dict | None = None):
        run.table.set_error(rowid, message, meta)
        run.phase_errors += 1
        run.phase_done += 1
        run.last_error = f"{run.table.get(rowid)['company_name']}: {message}"
        state["consecutive_failures"] += 1
        if state["consecutive_failures"] >= CIRCUIT_BREAKER and not state["fatal"]:
            state["fatal"] = f"Stopped after {CIRCUIT_BREAKER} failures in a row. Last error: {message}"
            run.cancel.set()

    async def _process(rowid: int, record: dict):
        params = build_request_params(
            record, cfg.model, cfg.task, list(cfg.rules), list(cfg.taxonomy),
            cfg.thinking_effort, system_prompt, schema,
        )
        chars = len(system_prompt) + len(params["messages"][0]["content"])
        if schema:
            chars += len(str(schema))
        chars4 = chars / 4

        start = time.time()
        for attempt in range(MAX_RETRIES + 1):
            if run.cancel.is_set():
                return
            est_in = chars4 * state["token_ratio"]
            ticket = await limiter.acquire(est_in, state["output_est"])
            retry_after = None
            try:
                raw = await client.messages.with_raw_response.create(**params)
                response = await raw.parse()
            except anthropic.APIStatusError as e:
                limiter.release(ticket, None, None)
                status = e.status_code
                if status in FATAL_STATUSES:
                    state["fatal"] = f"API error {status}: {e.message}"
                    run.cancel.set()
                    return
                if status not in RETRYABLE_STATUSES:
                    _row_failed(rowid, f"API error {status}: {e.message}")
                    return
                retry_after = _retry_after(e)
                if status == 429:
                    limiter.on_rate_limited(retry_after)
                last = f"API error {status}: {e.message}"
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                limiter.release(ticket, None, None)
                last = f"{type(e).__name__}: {e}"
            else:
                usage = response.usage
                in_tokens = (
                    usage.input_tokens
                    + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                )
                limiter.release(ticket, in_tokens, usage.output_tokens)
                if limiter.update_from_headers(raw.headers):
                    run.limits = {**limiter.limits(), "source": "your API key"}
                state["token_ratio"] = 0.9 * state["token_ratio"] + 0.1 * (in_tokens / max(chars4, 1))
                state["output_est"] = 0.9 * state["output_est"] + 0.1 * usage.output_tokens
                _handle_response(rowid, response, in_tokens, start)
                return

            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff_seconds(attempt, retry_after))
        _row_failed(rowid, f"Failed after {MAX_RETRIES} retries: {last}")

    def _handle_response(rowid: int, response, in_tokens: int, start: float):
        meta = {
            "model_used": response.model,
            "input_tokens": in_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": round(estimate_single_cost(in_tokens, response.usage.output_tokens, cfg.model), 6),
            "latency_ms": int((time.time() - start) * 1000),
        }
        if response.stop_reason == "max_tokens":
            _row_failed(rowid, "Response truncated (hit max_tokens)", meta)
            return
        if response.stop_reason == "refusal":
            _row_failed(rowid, "Model refused to answer", meta)
            return
        content_text, _ = extract_text(response)
        result = _parse_json_response(content_text)
        if not isinstance(result, dict):
            _row_failed(rowid, "Response was not valid JSON", meta)
            return
        run.table.set_result(rowid, result, meta)
        run.phase_done += 1
        state["consecutive_failures"] = 0

    async def _worker():
        while True:
            item = await queue.get()
            if item is None:
                return
            rowid, record = item
            if run.cancel.is_set():
                continue
            err = _integrity_error(run.table.get(rowid), record)
            if err:
                run.table.set_error(rowid, err)
                run.phase_errors += 1
                run.phase_done += 1
                continue
            await _process(rowid, record)

    producer = asyncio.create_task(_produce(rowids, queue, run, concurrency))
    await asyncio.gather(*[_worker() for _ in range(concurrency)])
    await producer
    await client.close()

    if state["fatal"]:
        raise FatalRunError(state["fatal"])
    if run.cancel.is_set():
        run._end("stopped", "Stopped by you")
        return False
    run._end("finished")
    return True


# ── Phase 2: Jev verification ──

async def _verify(run: BatchRun, jev_api_key: str, api_key: str | None):
    cfg = run.config
    codes = cfg.taxonomy_codes
    rules = list(cfg.rules)
    rowids = run.table.rowids_for_jev()
    run._begin("verifying", len(rowids))
    if not rowids:
        run._end("finished")
        return

    concurrency = 40
    limiter = RateLimiter.for_jev(max_concurrency=concurrency)
    run.limiter = limiter
    retry = RetryPolicy(max_retries=MAX_RETRIES, backoff_max=60.0, timeout=300.0)
    anthropic_client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=MAX_RETRIES) if api_key else None
    fatal = {"msg": None}
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(2 * concurrency, FETCH_CHUNK))

    async with AsyncTypeSafeClient(api_key=jev_api_key, retry=retry) as jev_client:

        async def _one(rowid: int, record: dict):
            row = run.table.get(rowid)
            result = {}
            for code in codes:
                result[code] = row[code]
                result[f"{code}_quote"] = row[f"{code}_quote"]
                result[f"{code}_supports"] = row[f"{code}_supports"]
            est = min(len(record["content"]) // 4, JEV_TOKEN_BUDGET) + 1_000
            ticket = await limiter.acquire(est)
            try:
                verification = await verify_extraction(
                    record, result, rules, codes,
                    api_key=jev_api_key,
                    anthropic_api_key=api_key,
                    jev_client=jev_client,
                    anthropic_client=anthropic_client,
                )
            except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as e:
                fatal["msg"] = f"Jev key rejected: {e}"
                run.cancel.set()
                return
            except anthropic.AuthenticationError as e:
                fatal["msg"] = f"Claude key rejected during section selection: {e}"
                run.cancel.set()
                return
            except Exception as e:
                run.table.set_jev_error(rowid, f"{type(e).__name__}: {e}")
                run.phase_errors += 1
                run.last_error = f"{row['company_name']}: {type(e).__name__}: {e}"
            else:
                run.table.set_jev(rowid, verification)
            finally:
                limiter.release(ticket, est, None)
            run.phase_done += 1

        async def _worker():
            while True:
                item = await queue.get()
                if item is None:
                    return
                rowid, record = item
                if run.cancel.is_set():
                    continue
                if record is None:
                    run.table.set_jev_error(rowid, "Row not found in database")
                    run.phase_errors += 1
                    run.phase_done += 1
                    continue
                await _one(rowid, record)

        producer = asyncio.create_task(_produce(rowids, queue, run, concurrency))
        await asyncio.gather(*[_worker() for _ in range(concurrency)])
        await producer

    if anthropic_client:
        await anthropic_client.close()

    if fatal["msg"]:
        raise FatalRunError(fatal["msg"])
    if run.cancel.is_set():
        run._end("stopped", "Stopped by you")
        return
    run._end("finished")
