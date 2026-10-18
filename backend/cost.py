from backend.templates import build_system_prompt, estimate_template_tokens

USD_TO_AUD = 1.55

PRICING = {
    "claude-fable-5-1": {
        "label": "Fable 5.1",
        "input": 10.0,
        "output": 50.0,
        "thinking": True,
    },
    "claude-opus-5-5": {
        "label": "Opus 5.5",
        "input": 4.0,
        "output": 20.0,
        "thinking": True,
    },
    "claude-sonnet-5-5": {
        "label": "Sonnet 5.5",
        "input": 2.0,
        "output": 10.0,
        "thinking": True,
    },
    "claude-sonnet-5": {
        "label": "Sonnet 5",
        "input": 2.0,
        "output": 10.0,
        "thinking": True,
    },
    "claude-haiku-5-5": {
        "label": "Haiku 5.5",
        "input": 0.10,
        "output": 0.50,
        "thinking": True,
        # Prompts over 100K input tokens are billed at the higher rate.
        "long_context": {"threshold": 100_000, "input": 0.50, "output": 2.50},
    },
}

DEFAULT_PRICING = {
    "label": "Unknown Model",
    "input": 3.0,
    "output": 15.0,
    "thinking": True,
}

MODEL_OPTIONS = [
    {"id": "claude-sonnet-5-5", "label": "Sonnet 5.5", "thinking": True},
    {"id": "claude-sonnet-5", "label": "Sonnet 5", "thinking": True},
    {"id": "claude-opus-5-5", "label": "Opus 5.5", "thinking": True},
    {"id": "claude-fable-5-1", "label": "Fable 5.1", "thinking": True},
    {"id": "claude-haiku-5-5", "label": "Haiku 5.5", "thinking": True},
]

DEFAULT_OUTPUT_TOKENS = 500

JEV_PRICING = {
    "input": 0.042,
    "output": 0.0,
}


def estimate_jev_cost(input_tokens: int) -> float:
    return input_tokens / 1_000_000 * JEV_PRICING["input"]


def match_pricing(model_id: str) -> dict:
    for family, pricing in PRICING.items():
        if model_id.startswith(family):
            return pricing
    return DEFAULT_PRICING


_EFFORT_TO_EST_TOKENS = {
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}


def estimate_cost(
    count: int,
    avg_content_length: float,
    model: str,
    task: str,
    rules: list[str] | None = None,
    taxonomy: list[dict] | None = None,
    thinking_effort: str = "high",
    output_tokens_estimate: int = DEFAULT_OUTPUT_TOKENS,
) -> dict:
    prices = match_pricing(model)

    system_tokens = len(build_system_prompt()) // 4
    template_tokens = estimate_template_tokens(task, rules, taxonomy)
    avg_content_tokens = int(avg_content_length / 4)
    per_record_input = avg_content_tokens + system_tokens + template_tokens

    thinking_per_record = _EFFORT_TO_EST_TOKENS.get(thinking_effort, 0) if prices["thinking"] else 0
    per_record_output = output_tokens_estimate + thinking_per_record

    total_input = per_record_input * count
    total_output = per_record_output * count

    input_cost = total_input / 1_000_000 * prices["input"]
    output_cost = total_output / 1_000_000 * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "record_count": count,
        "avg_input_tokens_per_record": per_record_input,
        "avg_content_tokens": avg_content_tokens,
        "system_prompt_tokens": system_tokens,
        "template_overhead_tokens": template_tokens,
        "estimated_output_tokens_per_record": per_record_output,
        "thinking_tokens_per_record": thinking_per_record,
        "model": model,
        "model_label": prices["label"],
        "input_rate": prices["input"],
        "output_rate": prices["output"],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 2),
    }


def estimate_single_cost(
    input_tokens: int, output_tokens: int, model: str,
) -> float:
    prices = match_pricing(model)
    long_context = prices.get("long_context")
    if long_context and input_tokens > long_context["threshold"]:
        prices = long_context
    return (
        input_tokens / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )


def estimate_full_run(
    count: int,
    avg_input_tokens: float,
    avg_output_tokens: float,
    model: str,
    avg_latency_s: float,
    concurrency: int,
    limits: dict,
    utilisation: float = 0.8,
) -> dict:
    """Cost and time range for `count` docs. `limits` holds per-minute rpm/itpm/otpm."""
    cost_usd = count * estimate_single_cost(avg_input_tokens, avg_output_tokens, model)

    bounds = {
        "requests": limits["rpm"] * utilisation,
        "input tokens": limits["itpm"] * utilisation / max(avg_input_tokens, 1),
        "output tokens": limits["otpm"] * utilisation / max(avg_output_tokens, 1),
        "concurrency": concurrency / max(avg_latency_s, 1) * 60,
    }
    bottleneck = min(bounds, key=bounds.get)
    docs_per_min = bounds[bottleneck]
    best_min = count / docs_per_min if docs_per_min > 0 else 0.0

    return {
        "cost_usd": round(cost_usd, 2),
        "docs_per_min": round(docs_per_min, 1),
        "bottleneck": bottleneck,
        "minutes_low": best_min * 0.85,
        "minutes_high": best_min * 1.55,
    }


def suggested_concurrency(
    avg_input_tokens: float,
    avg_output_tokens: float,
    avg_latency_s: float,
    limits: dict,
    utilisation: float = 0.8,
) -> int:
    docs_per_min = min(
        limits["rpm"] * utilisation,
        limits["itpm"] * utilisation / max(avg_input_tokens, 1),
        limits["otpm"] * utilisation / max(avg_output_tokens, 1),
    )
    in_flight = docs_per_min / 60 * avg_latency_s * 1.2
    return int(max(5, min(100, round(in_flight))))
