import asyncio
import json
import re

import anthropic
from typesafe_sdk import AsyncTypeSafeClient, Noul

JEV_TOKEN_BUDGET = 28_000

SECTION_SELECTOR_PROMPT = """\
You are given a financial report document and a set of extracted data with \
supporting quotes. The document is divided into numbered sections at '#' \
heading boundaries. Each section starts with a marker like \
"=== SECTION [0]: heading text ===".

Identify which sections contain:
- The source material where the extracted quotes came from
- Context needed to verify the values (statement headers, units, column labels)

Err on the side of inclusion — better to select extra sections than miss one.
Select between 3 and 20 sections, ordered by relevance (most important first).

# Company
{company_name} — {year} — {section_name}

# Extraction Results
{formatted_extraction}

# Rules
{formatted_rules}

# Document
{formatted_document}
"""

SECTION_SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sections"],
    "additionalProperties": False,
}


def build_verification_nouls(
    rules: list[str],
    result: dict,
    taxonomy_codes: list[str],
) -> dict[str, Noul]:
    nouls = {}

    field_values = {code: result[code] for code in taxonomy_codes}

    for i, rule_text in enumerate(rules):
        nouls[f"rule_{i + 1}"] = Noul(
            instructions={
                "check": "Did the result violate this rule?",
                "rule": rule_text,
                "result": field_values,
            },
        )

    for code in taxonomy_codes:
        if result[code] == "Don't Know":
            nouls[f"{code}::dont_know_validation"] = Noul(
                instructions={
                    "check": (
                        "Does the document contain enough information "
                        "to determine a value for this field?"
                    ),
                    "field": code,
                },
            )
            continue

        quote = result.get(f"{code}_quote", "")

        nouls[f"{code}::quote_fabrication"] = Noul(
            instructions={
                "check": "Are any of the quotes not found in the document?",
                "quotes": quote,
            },
        )

    for code in taxonomy_codes:
        if result[code] == "Don't Know":
            continue

        quote = result.get(f"{code}_quote", "")
        supports = result.get(f"{code}_supports", "")

        nouls[f"{code}::argument_support"] = Noul(
            instructions={
                "check": "Do the quotes and argument fail to support the chosen value?",
                "value": result[code],
                "quotes": quote,
                "argument": supports,
            },
        )

    return nouls


def build_jev_state(record: dict, result: dict, taxonomy_codes: list[str]) -> dict:
    state = {
        "document": record["content"],
        "result": {},
        "context": {
            "company": record["company_name"],
            "year": record["year"],
            "section": record["section_name"],
        },
    }
    for code in taxonomy_codes:
        state["result"][code] = result[code]
        state["result"][f"{code}_quote"] = result.get(f"{code}_quote", "")
        state["result"][f"{code}_supports"] = result.get(f"{code}_supports", "")
    return state


def extract_markdown_sections(content: str, level: str = "#") -> list[dict]:
    pattern = re.compile(r"^" + re.escape(level) + r" ", re.MULTILINE)
    lines = content.split("\n")
    boundaries = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            boundaries.append(i)

    if not boundaries:
        return []

    sections = []
    if boundaries[0] > 0:
        preamble = "\n".join(lines[:boundaries[0]])
        sections.append({
            "heading": "(preamble)",
            "content": preamble,
            "index": 0,
            "estimated_tokens": len(preamble) // 4,
        })

    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(lines)
        heading = lines[start].lstrip("# ").strip()
        section_text = "\n".join(lines[start:end])
        sections.append({
            "heading": heading,
            "content": section_text,
            "index": len(sections),
            "estimated_tokens": len(section_text) // 4,
        })

    return sections


def _build_marked_document(sections: list[dict]) -> str:
    parts = []
    for s in sections:
        parts.append(f"=== SECTION [{s['index']}]: {s['heading']} ===")
        parts.append(s["content"])
    return "\n".join(parts)


def _format_extraction(result: dict, taxonomy_codes: list[str]) -> str:
    lines = []
    for code in taxonomy_codes:
        val = result.get(code, "")
        quote = result.get(f"{code}_quote", "")
        supports = result.get(f"{code}_supports", "")
        if val == "Don't Know":
            lines.append(f"Field: {code} = \"Don't Know\"")
        else:
            lines.append(f"Field: {code} = \"{val}\"")
            if quote:
                lines.append(f"  Quote: {quote}")
            if supports:
                lines.append(f"  Supports: {supports}")
    return "\n".join(lines)


async def select_relevant_sections(
    sections: list[dict],
    result: dict,
    taxonomy_codes: list[str],
    rules: list[str],
    record: dict,
    *,
    api_key: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> list[dict]:
    client = client or anthropic.AsyncAnthropic(api_key=api_key)

    formatted_document = _build_marked_document(sections)
    formatted_extraction = _format_extraction(result, taxonomy_codes)
    rules_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules)) if rules else "(no rules)"

    user_msg = SECTION_SELECTOR_PROMPT.format(
        company_name=record["company_name"],
        year=record["year"],
        section_name=record["section_name"],
        formatted_extraction=formatted_extraction,
        formatted_rules=rules_text,
        formatted_document=formatted_document,
    )

    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=16_000,
        messages=[{"role": "user", "content": user_msg}],
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": SECTION_SELECTOR_SCHEMA},
        },
    )

    content_text = ""
    for block in response.content:
        if block.type == "text":
            content_text = block.text

    return json.loads(content_text)["sections"]


def assemble_trimmed_document(
    sections: list[dict],
    selected: list[dict],
    budget: int = JEV_TOKEN_BUDGET,
) -> tuple[str, list[int]]:
    has_preamble = sections and sections[0]["heading"] == "(preamble)"
    must_keep = {0} if has_preamble else set()
    priority_order = [s["index"] for s in selected]

    ordered_indices = list(must_keep)
    for idx in priority_order:
        if idx not in must_keep and 0 <= idx < len(sections):
            ordered_indices.append(idx)

    kept = []
    running_tokens = 0
    for idx in ordered_indices:
        section = sections[idx]
        if running_tokens + section["estimated_tokens"] > budget:
            continue
        kept.append(idx)
        running_tokens += section["estimated_tokens"]

    kept.sort()

    trimmed_parts = [sections[idx]["content"] for idx in kept]
    trimmed_content = "\n\n".join(trimmed_parts)
    return trimmed_content, kept


def compute_field_confidence(
    field_code: str,
    jev_answers: dict,
) -> dict:
    checks = []

    dk_key = f"{field_code}::dont_know_validation"
    if dk_key in jev_answers:
        checks.append({
            "check": dk_key,
            "p_wrong": round(jev_answers[dk_key].noul, 3),
            "label": f"{field_code} don't know validation",
        })

    fab_key = f"{field_code}::quote_fabrication"
    if fab_key in jev_answers:
        checks.append({
            "check": fab_key,
            "p_wrong": round(jev_answers[fab_key].noul, 3),
            "label": f"{field_code} quote fabrication",
        })

    arg_key = f"{field_code}::argument_support"
    if arg_key in jev_answers:
        checks.append({
            "check": arg_key,
            "p_wrong": round(jev_answers[arg_key].noul, 3),
            "label": f"{field_code} argument support",
        })

    max_p_wrong = max((c["p_wrong"] for c in checks), default=0.0)

    return {
        "score": round(1.0 - max_p_wrong, 3),
        "checks": checks,
    }


def compute_dn_confidence(field_confidences: dict) -> float | None:
    dk_scores = []
    for fc_data in field_confidences.values():
        for check in fc_data.get("checks", []):
            if "dont_know_validation" in check.get("check", ""):
                dk_scores.append(1.0 - check["p_wrong"])
    return round(min(dk_scores), 3) if dk_scores else None


def compute_record_confidence(field_confidences: dict) -> float:
    scores = [fc["score"] for fc in field_confidences.values()]
    return min(scores) if scores else 0.0


def compute_battery_confidences(jev_answers: dict, rules: list[str], taxonomy_codes: list[str]) -> dict:
    rule_checks = []
    rule_p = []
    for i, rule_text in enumerate(rules):
        key = f"rule_{i + 1}"
        if key in jev_answers:
            p = round(jev_answers[key].noul, 3)
            rule_p.append(p)
            rule_checks.append({
                "rule": rule_text,
                "p_wrong": p,
            })

    quote_p = []
    for code in taxonomy_codes:
        key = f"{code}::quote_fabrication"
        if key in jev_answers:
            quote_p.append(jev_answers[key].noul)

    reason_p = []
    for code in taxonomy_codes:
        key = f"{code}::argument_support"
        if key in jev_answers:
            reason_p.append(jev_answers[key].noul)

    return {
        "confidence_rule": round(1.0 - max(rule_p), 3) if rule_p else None,
        "rule_checks": rule_checks,
        "confidence_quote": round(1.0 - max(quote_p), 3) if quote_p else None,
        "confidence_reason": round(1.0 - max(reason_p), 3) if reason_p else None,
    }


async def verify_extraction(
    record: dict,
    result: dict,
    rules: list[str],
    taxonomy_codes: list[str],
    *,
    api_key: str,
    anthropic_api_key: str | None = None,
    jev_client: AsyncTypeSafeClient | None = None,
    anthropic_client: anthropic.AsyncAnthropic | None = None,
) -> dict:
    content = record["content"]
    estimated_tokens = len(content) // 4
    trim_meta = {}

    if estimated_tokens > JEV_TOKEN_BUDGET:
        sections = extract_markdown_sections(content, "#")
        if len(sections) < 3:
            sections = extract_markdown_sections(content, "##")
        if len(sections) < 3:
            return {
                "skipped": True,
                "skip_reason": "document_too_large_no_headings",
                "document_tokens_estimated": estimated_tokens,
            }
        if not anthropic_api_key and anthropic_client is None:
            return {
                "skipped": True,
                "skip_reason": "document_too_large_no_anthropic_key",
                "document_tokens_estimated": estimated_tokens,
            }

        selected = await select_relevant_sections(
            sections, result, taxonomy_codes, rules, record,
            api_key=anthropic_api_key,
            client=anthropic_client,
        )

        if not selected:
            return {
                "skipped": True,
                "skip_reason": "no_sections_selected",
                "document_tokens_estimated": estimated_tokens,
            }

        trimmed_content, kept_indices = assemble_trimmed_document(sections, selected)

        if not kept_indices:
            return {
                "skipped": True,
                "skip_reason": "document_too_large_after_trimming",
                "document_tokens_estimated": estimated_tokens,
            }

        trimmed_record = {**record, "content": trimmed_content}
        state = build_jev_state(trimmed_record, result, taxonomy_codes)
        trim_meta = {
            "trimmed": True,
            "sections_kept": kept_indices,
            "sections_total": len(sections),
            "document_tokens_original": estimated_tokens,
            "document_tokens_trimmed": len(trimmed_content) // 4,
        }
    else:
        state = build_jev_state(record, result, taxonomy_codes)

    questions = build_verification_nouls(rules, result, taxonomy_codes)

    if jev_client is not None:
        response = await jev_client.system_one(state=state, questions=questions)
    else:
        async with AsyncTypeSafeClient(api_key=api_key) as client:
            response = await client.system_one(
                state=state,
                questions=questions,
            )

    jev_answers = response.answers

    field_confidences = {}
    for code in taxonomy_codes:
        field_confidences[code] = compute_field_confidence(code, jev_answers)

    record_confidence = compute_record_confidence(field_confidences)
    battery_confidences = compute_battery_confidences(jev_answers, rules, taxonomy_codes)

    return {
        "field_confidences": field_confidences,
        "record_confidence": record_confidence,
        **battery_confidences,
        **trim_meta,
    }


async def verify_trial_results(
    samples: list[dict],
    trial_results: list[dict],
    rules: list[str],
    taxonomy_codes: list[str],
    *,
    api_key: str,
    anthropic_api_key: str | None = None,
) -> list[dict]:
    semaphore = asyncio.Semaphore(5)

    async def _verify_one(record: dict, trial_result: dict) -> dict | None:
        if trial_result["status"] != "success":
            return None
        if not isinstance(trial_result.get("result"), dict):
            return None
        async with semaphore:
            try:
                return await verify_extraction(
                    record,
                    trial_result["result"],
                    rules,
                    taxonomy_codes,
                    api_key=api_key,
                    anthropic_api_key=anthropic_api_key,
                )
            except Exception:
                return None

    tasks = [
        _verify_one(sample, trial_result)
        for sample, trial_result in zip(samples, trial_results)
    ]
    return await asyncio.gather(*tasks)


def validate_jev_api_key(api_key: str) -> bool:
    async def _validate():
        try:
            async with AsyncTypeSafeClient(api_key=api_key) as client:
                await client.system_one(
                    state={"test": "validation"},
                    questions={"ping": Noul(
                        instructions={"check": "Is this a test?"},
                    )},
                )
            return True
        except Exception:
            return False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_validate())
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, _validate()).result()
