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
    semaphore = asyncio.Semaphore(5)
    system_prompt = build_system_prompt()
    schema = build_json_schema(taxonomy) if taxonomy else None

    FALLBACK_MODEL = "claude-sonnet-5"

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
                if model != FALLBACK_MODEL:
                    try:
                        result = await _call_model(sample, FALLBACK_MODEL)
                    except Exception as retry_e:
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
                            "model_used": FALLBACK_MODEL,
                            "status": "error",
                            "error": f"Fallback ({FALLBACK_MODEL}) also failed: {retry_e}",
                            "latency_ms": int((time.time() - start) * 1000),
                        }
                else:
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
                        "error": str(e),
                        "latency_ms": int((time.time() - start) * 1000),
                    }
            except Exception as e:
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
                    "error": str(e),
                    "latency_ms": int((time.time() - start) * 1000),
                }

            response = result["response"]
            result_data = _parse_json_response(result["content_text"])

            return {
                "rowid": sample["rowid"],
                "company_name": sample["company_name"],
                "year": sample["year"],
                "section_name": sample["section_name"],
                "country": sample.get("country", ""),
                "source_text": sample["content"],
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "result": result_data,
                "raw_response": result["content_text"],
                "thinking_text": result["thinking_text"],
                "model_used": result["model_used"],
                "status": "success",
                "error": None,
                "latency_ms": int((time.time() - start) * 1000),
            }

    results = await asyncio.gather(*[process_one(s) for s in samples])

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

    total_cost = sum(
        estimate_single_cost(r["input_tokens"], r["output_tokens"], r.get("model_used", model))
        for r in results
        if r["status"] == "success"
    )

    return {
        "trial_id": trial_id,
        "samples": list(results),
        "total_cost_usd": round(total_cost, 6),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
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
