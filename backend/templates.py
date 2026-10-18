import textwrap

SYSTEM_PROMPT = (
    "You are a financial data extraction and classifier specialist. You will be "
    "given a section from a company's annual financial report. Extract or classify "
    "the requested information precisely from the provided text."
)

# Prompt structure reference (not used at runtime — see build_user_message()).
# The system message (SYSTEM_PROMPT) sets the model's role separately.
# Rules section is optional — only included when provided.
# Taxonomy is conveyed via the JSON schema, not repeated in the prompt.
PROMPT_TEMPLATE = textwrap.dedent("""\
    # Context
    Inside the <document> tags are the {section} from the {year} annual report
    of the ASX-listed company: {company}.

    # Task
    Please complete the following task(s) based on the provided document.
    Ensure you base all answers STRICTLY on the text provided in the
    <document> section. Do not use outside knowledge.

    {task}

    # Rules
    Follow these rules strictly when completing the task above. Each rule must
    be obeyed.
    1. {rule_1}
    2. {rule_2}
    ...

    # Document
    <document>
    {content}
    </document>""")

RULES_PREAMBLE = (
    "Follow these rules strictly when completing the task above. "
    "Each rule must be obeyed — do not deviate."
)


def _format_rules(rules: list[str]) -> str:
    if not rules:
        return ""
    numbered = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(rules))
    return f"# Rules\n{RULES_PREAMBLE}\n{numbered}"



def build_json_schema(taxonomy: list[dict]) -> dict:
    properties = {}
    required = []
    for entry in taxonomy:
        prop = {"type": "string"}
        if entry.get("categories"):
            prop["enum"] = [cat["name"] for cat in entry["categories"]] + ["Don't Know"]
            cat_descriptions = ". ".join(
                f"{cat['name']}: {cat['description']}"
                for cat in entry["categories"]
            )
            prop["description"] = (
                f"{entry.get('definition', '')}. {cat_descriptions}. "
                "Don't Know: Use when the document does not contain enough "
                "information to determine a value for this field."
            )
        else:
            prop["description"] = (
                f"{entry.get('definition', '')} "
                "If the information cannot be found or the document lacks the "
                "necessary information for a decision, respond with exactly "
                "\"Don't Know\"."
            )
        properties[entry["code"]] = prop
        required.append(entry["code"])
        quote_key = f"{entry['code']}_quote"
        properties[quote_key] = {
            "type": "string",
            "description": (
                f"One or more exact facts and quotes from the document supporting "
                f"the value for '{entry['code']}'. Number each quote "
                f"(e.g. '1. ... 2. ...'). Do not paraphrase or stitch text "
                f"from different locations."
            ),
        }
        required.append(quote_key)

        supports_key = f"{entry['code']}_supports"
        properties[supports_key] = {
            "type": "string",
            "description": (
                f"Explain how the quotes above support the chosen value "
                f"for '{entry['code']}'."
            ),
        }
        required.append(supports_key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_message(
    company: str, year: str, section: str, content: str,
    task: str, rules: list[str] | None = None,
    taxonomy: list[dict] | None = None,
    section_description: str | None = None,
) -> str:
    context = (
        f"# Context\n"
        f"Inside the <document> tags are the {section} from the "
        f"{year} annual report of the ASX-listed company: {company}."
    )
    if section_description:
        context += f"\nThe {section} consists of: {section_description}."
    parts = [
        context,
        (
            "# Task\n"
            "Please complete the following task(s) based on the provided document. "
            "Ensure you base all answers STRICTLY on the text provided in the "
            "<document> section. Do not use outside knowledge.\n\n"
            f"{task}"
        ),
    ]
    rules_block = _format_rules(rules or [])
    if rules_block:
        parts.append(rules_block)
    parts.append(f"# Document\n<document>\n{content}\n</document>")
    return "\n\n".join(parts)


def estimate_template_tokens(
    task: str, rules: list[str] | None = None,
    taxonomy: list[dict] | None = None,  # noqa: ARG001
) -> int:
    rules_text = _format_rules(rules or [])
    overhead = len(SYSTEM_PROMPT) + len(task) + len(rules_text) + 200
    return overhead // 4
