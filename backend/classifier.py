import asyncio
import json
import time
from uuid import uuid4

import anthropic

from backend.cost import MODEL_OPTIONS, estimate_single_cost
from backend.database import SECTION_DESCRIPTIONS
from backend.templates import build_json_schema, build_system_prompt, build_user_message

_ADAPTIVE_PATTERNS = (
    "opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "opus-5-5",
    "sonnet-4-6", "sonnet-5", "sonnet-5-5",
    "fable-5", "fable-5-1",
    "haiku-5-5",
    "mythos-5", "mythos-5-1", "mythos-preview",
)

_EFFORT_TO_BUDGET = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}

_EFFORT_TO_MAX_TOKENS = {
    "low": 8192,
    "medium": 16000,
    "high": 16000,
    "xhigh": 64000,
    "max": 64000,
}


def _uses_adaptive(model: str) -> bool:
    return any(p in model for p in _ADAPTIVE_PATTERNS)


def _build_thinking_params(model: str, effort: str) -> tuple[dict, dict | None, int]:
    max_tokens = _EFFORT_TO_MAX_TOKENS.get(effort, 16000)
    if _uses_adaptive(model):
        return {"type": "adaptive"}, {"effort": effort}, max_tokens
    budget = _EFFORT_TO_BUDGET.get(effort, 8192)
    return {"type": "enabled", "budget_tokens": budget}, None, max_tokens


def validate_api_key(api_key: str) -> dict:
    if not api_key:
        return {"valid": False, "message": "No API key provided."}

    client = anthropic.Anthropic(api_key=api_key)

    try:
        client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=10,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Hi"}],
        )
    except anthropic.AuthenticationError:
        return {"valid": False, "message": "Invalid API key."}
    except anthropic.BadRequestError as e:
        if "anthropic-workspace-id" in str(e):
            return {
                "valid": False,
                "message": (
                    "Error code: 400 — This API key is likely an Organisation API key "
                    "and not a Workspace API key. Please create and use a Workspace API key."
                ),
            }
        return {"valid": False, "message": f"Validation error: {e}"}
    except Exception as e:
        return {"valid": False, "message": f"Validation error: {type(e).__name__}: {e}"}

    return {"valid": True, "message": "API key is valid.", "models": MODEL_OPTIONS}


def _parse_json_response(text: str) -> dict | str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def build_request_params(
    sample: dict,
    model: str,
    task: str,
    rules: list[str] | None,
    taxonomy: list[dict] | None,
    thinking_effort: str,
    system_prompt: str,
    schema: dict | None,
) -> dict:
    user_msg = build_user_message(
        sample["company_name"],
        sample["year"],
        sample["section_name"],
        sample["content"],
        task,
        rules,
        taxonomy,
        section_description=SECTION_DESCRIPTIONS.get(sample["section_name"]),
    )
    thinking, output_cfg, max_tokens = _build_thinking_params(model, thinking_effort)
    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
        "thinking": thinking,
    }
    if schema:
        output_cfg = output_cfg or {}
        output_cfg["format"] = {
            "type": "json_schema",
            "schema": schema,
        }
    if output_cfg:
        params["output_config"] = output_cfg
    return params


def extract_text(response) -> tuple[str, str]:
    content_text = ""
    thinking_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            content_text = block.text
    return content_text, thinking_text


FALLBACK_MODEL = "claude-sonnet-5"


def _error_result(sample: dict, model: str, message: str, start: float) -> dict:
    return {
        "rowid": sample["rowid"],
        "company_name": sample["company_name"],
        "year": sample["year"],
        "section_name": sample["section_name"],
        "country": sample.get("country", ""),
        "source_text": sample["content"],
        "input_tokens": 0,
        "output_tokens": 0,
        "result": None,
        "raw_response": None,
        "model_used": model,
        "status": "error",
        "error": message,
        "latency_ms": int((time.time() - start) * 1000),
    }


async def _classify_many(
    client,
    samples: list[dict],
    model: str,
    task: str,
    rules: list[str] | None,
    taxonomy: list[dict] | None,
    thinking_effort: str,
    system_prompt: str,
    schema: dict | None,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Classify the samples at the same time (limited by the semaphore). Keeps input order."""

    async def _call_model(sample: dict, use_model: str) -> dict:
        params = build_request_params(
            sample, use_model, task, rules, taxonomy,
            thinking_effort, system_prompt, schema,
        )
        response = await client.messages.create(**params)
        content_text, thinking_text = extract_text(response)
        return {
            "response": response,
            "content_text": content_text,
            "thinking_text": thinking_text,
            "model_used": use_model,
        }

    async def process_one(sample: dict) -> dict:
        async with semaphore:
            start = time.time()
            try:
                result = await _call_model(sample, model)
            except anthropic.BadRequestError as e:
                if model == FALLBACK_MODEL:
                    return _error_result(sample, model, str(e), start)
                try:
                    result = await _call_model(sample, FALLBACK_MODEL)
                except Exception as retry_e:
                    return _error_result(
                        sample, FALLBACK_MODEL,
                        f"Fallback ({FALLBACK_MODEL}) also failed: {retry_e}", start,
                    )
            except Exception as e:
                return _error_result(sample, model, str(e), start)

            response = result["response"]
            return {
                "rowid": sample["rowid"],
                "company_name": sample["company_name"],
                "year": sample["year"],
                "section_name": sample["section_name"],
                "country": sample.get("country", ""),
                "source_text": sample["content"],
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "result": _parse_json_response(result["content_text"]),
                "raw_response": result["content_text"],
                "thinking_text": result["thinking_text"],
                "model_used": result["model_used"],
                "status": "success",
                "error": None,
                "latency_ms": int((time.time() - start) * 1000),
            }

    return list(await asyncio.gather(*[process_one(s) for s in samples]))


def _total_cost(results: list[dict], model: str) -> float:
    return round(sum(
        estimate_single_cost(r["input_tokens"], r["output_tokens"], r.get("model_used", model))
        for r in results
        if r["status"] == "success"
    ), 6)


async def run_trial(
    samples: list[dict],
    model: str,
    task: str,
    rules: list[str] | None = None,
    taxonomy: list[dict] | None = None,
    thinking_effort: str = "high",
    *,
    api_key: str,
    trial_store: dict,
) -> dict:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    schema = build_json_schema(taxonomy) if taxonomy else None
    results = await _classify_many(
        client, samples, model, task, rules, taxonomy, thinking_effort,
        build_system_prompt(), schema, asyncio.Semaphore(5),
    )

    trial_id = f"trial_{uuid4().hex[:12]}"
    trial_store[trial_id] = {
        "samples": list(results),
        "model": model,
        "task": task,
        "rules": rules,
        "taxonomy": taxonomy,
        "thinking_effort": thinking_effort,
        "raw_samples": samples,
    }

    return {
        "trial_id": trial_id,
        "samples": list(results),
        "total_cost_usd": _total_cost(results, model),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
    }


# ── Targeted sample (top-N keyword search) ──

SEARCH_TERMS_MODEL = "claude-haiku-5-5"
TARGETED_GROUP_SIZE = 5

STOP_FOUND_ALL = "found all hits"
STOP_MAX_REPORTS = "reached the maximum number of reports"
STOP_NO_MORE = "no more matching reports"

_SEARCH_TERMS_SCHEMA = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["terms"],
    "additionalProperties": False,
}


async def make_search_terms(task: str, field: dict, values: list[str], *, api_key: str) -> list[str]:
    """Keywords for the BM25 search. The hit values are always included.
    If the call fails, the hit values and the field name are the only terms."""
    fallback = list(values) + [field["code"].replace("_", " ")]
    cats = {c["name"]: c.get("description", "") for c in field.get("categories", [])}
    if values:
        wanted = "\n".join(f"- {v}" + (f": {cats[v]}" if cats.get(v) else "") for v in values)
    else:
        wanted = "- Any report that gives a clear answer for this field."
    prompt = (
        "We search sections of ASX/NZX company annual reports with a keyword (BM25) search. "
        "We want the reports that are most likely to match the description below.\n\n"
        f"Task given to the classifier:\n{task}\n\n"
        f"Field: {field['code']}\n"
        f"Field definition: {field.get('definition', '')}\n\n"
        f"Values we want to find:\n{wanted}\n\n"
        "Give 10 to 30 search terms that such a report is likely to contain: words and short "
        "phrases (1 to 4 words) with the same meaning, short forms, and closely related words. "
        "Use plain words only, with no search operators. Do not include words that almost "
        "every annual report contains."
    )
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=SEARCH_TERMS_MODEL,
            max_tokens=2000,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _SEARCH_TERMS_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        content_text, _ = extract_text(response)
        data = _parse_json_response(content_text)
        terms = data.get("terms", []) if isinstance(data, dict) else []
        terms = [str(t) for t in terms if str(t).strip()]
    except Exception:
        terms = []
    return list(values) + terms if terms else fallback


def is_hit(result: dict, field: str, values: list[str]) -> bool:
    """A hit = a successful result whose field has one of the values. With no values
    (a field with no categories), any answer except Don't Know or empty is a hit."""
    if result.get("status") != "success" or not isinstance(result.get("result"), dict):
        return False
    answer = str(result["result"].get(field, "") or "").strip()
    if values:
        return answer in values
    return answer not in ("", "Don't Know")


async def run_targeted(
    ranked: list[dict],
    hit_field: str,
    hit_values: list[str],
    n_hits: int,
    max_reports: int,
    model: str,
    task: str,
    rules: list[str] | None = None,
    taxonomy: list[dict] | None = None,
    thinking_effort: str = "high",
    *,
    api_key: str,
    trial_store: dict,
    get_records,
    skip_rowids=(),
) -> dict:
    """Classify the ranked reports in groups, best match first. Stop when n_hits hits are
    found, max_reports reports are read, or no ranked reports are left.

    get_records(rowids) returns {rowid: record with content}. The trial_store entry and
    the returned "samples" hold the hits only. "all_results" holds every report read."""
    client = anthropic.AsyncAnthropic(api_key=api_key)
    schema = build_json_schema(taxonomy) if taxonomy else None
    system_prompt = build_system_prompt()
    semaphore = asyncio.Semaphore(TARGETED_GROUP_SIZE)

    skip = set(skip_rowids)
    queue = [r for r in ranked if r["rowid"] not in skip]
    hits, hit_records, all_results, read_rowids = [], [], [], []
    pos = 0
    while True:
        if len(hits) >= n_hits:
            stop_reason = STOP_FOUND_ALL
            break
        if len(all_results) >= max_reports:
            stop_reason = STOP_MAX_REPORTS
            break
        if pos >= len(queue):
            stop_reason = STOP_NO_MORE
            break
        group = queue[pos:pos + min(TARGETED_GROUP_SIZE, max_reports - len(all_results))]
        pos += len(group)
        rowids = [r["rowid"] for r in group]
        read_rowids += rowids
        records_by_id = await asyncio.to_thread(get_records, rowids)
        records = [records_by_id[rid] for rid in rowids if rid in records_by_id]
        if not records:
            continue
        results = await _classify_many(
            client, records, model, task, rules, taxonomy, thinking_effort,
            system_prompt, schema, semaphore,
        )
        for record, res in zip(records, results):
            all_results.append(res)
            if is_hit(res, hit_field, hit_values):
                hits.append(res)
                hit_records.append(record)

    trial_id = f"trial_{uuid4().hex[:12]}"
    trial_store[trial_id] = {
        "samples": list(hits),
        "model": model,
        "task": task,
        "rules": rules,
        "taxonomy": taxonomy,
        "thinking_effort": thinking_effort,
        "raw_samples": hit_records,
        "mode": "Targeted",
        "read_rowids": list(skip_rowids) + read_rowids,
    }

    # Misses are used only for token counts, so drop their text to keep the session small.
    hit_ids = {id(h) for h in hits}
    slim = [
        r if id(r) in hit_ids else {**r, "source_text": "", "thinking_text": "", "raw_response": None}
        for r in all_results
    ]
    return {
        "trial_id": trial_id,
        "samples": list(hits),
        "total_cost_usd": _total_cost(all_results, model),
        "success_count": len(hits),
        "error_count": sum(1 for r in all_results if r["status"] == "error"),
        "all_results": slim,
        "reports_read": len(all_results),
        "hits_found": len(hits),
        "stop_reason": stop_reason,
    }


async def retry_trial_sample(
    trial_id: str,
    sample_index: int,
    new_task: str | None = None,
    new_rules: list[str] | None = None,
    new_taxonomy: list[dict] | None = None,
    new_sample: dict | None = None,
    *,
    api_key: str,
    trial_store: dict,
) -> dict:
    trial = trial_store.get(trial_id)
    if not trial:
        return {"error": True, "message": f"Trial {trial_id} not found."}

    if sample_index < 0 or sample_index >= len(trial["raw_samples"]):
        return {"error": True, "message": "Invalid sample index."}

    sample = new_sample if new_sample else trial["raw_samples"][sample_index]
    task = new_task if new_task is not None else trial["task"]
    rules = new_rules if new_rules is not None else trial.get("rules")
    taxonomy = new_taxonomy if new_taxonomy is not None else trial.get("taxonomy")

    result = await run_trial(
        [sample], trial["model"], task, rules,
        taxonomy=taxonomy,
        thinking_effort=trial.get("thinking_effort", "high"),
        api_key=api_key, trial_store=trial_store,
    )

    if result.get("error"):
        return result

    new_result = result["samples"][0]
    trial["samples"][sample_index] = new_result
    if new_sample:
        trial["raw_samples"][sample_index] = new_sample

    return {"sample": new_result}
