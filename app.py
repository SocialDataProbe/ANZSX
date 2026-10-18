import asyncio
import concurrent.futures
import traceback

import streamlit as st

from backend import batch_processor, classifier, database
from backend.batch_table import BatchTable, RunConfig
from backend.cost import estimate_cost, estimate_full_run, match_pricing, suggested_concurrency, USD_TO_AUD
from backend.rate_limiter import start_tier_limits
from backend.templates import build_user_message
from backend.verification import compute_dn_confidence, validate_jev_api_key, verify_trial_results

st.set_page_config(
    page_title="ASX/NZX Annual Reports Data Extraction App",
    layout="wide",
    page_icon="https://about.unimelb.edu.au/__data/assets/image/0024/1885839/uom-logo.png",
)

# ── University of Melbourne branding ──────────────────────────────

_UOM_FONT = '<style>@import url("https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap");</style>'

_UOM_CSS = """
<style>

:root {
    --uom-blue: #094183;
    --uom-blue-light: #0E6EB8;
    --uom-blue-pale: #E8EEF7;
    --uom-white: #FFFFFF;
    --uom-grey-light: #F0F2F6;
    --uom-grey-mid: #D4D4D4;
    --uom-text: #2C2C2C;
}

/* Font override */
html, body, [class*="st-"], .stMarkdown, .stTextInput label,
.stSelectbox label, .stMultiSelect label, .stTextArea label {
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif !important;
}

/* Top header bar */
header[data-testid="stHeader"] {
    background-color: var(--uom-blue) !important;
}
header[data-testid="stHeader"] * {
    color: var(--uom-white) !important;
}
header[data-testid="stHeader"] button {
    color: var(--uom-white) !important;
}
header[data-testid="stHeader"] svg {
    fill: var(--uom-white) !important;
    stroke: var(--uom-white) !important;
}

/* Main title */
h1 {
    color: var(--uom-blue) !important;
    font-weight: 700 !important;
    font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Tab styling */
button[data-baseweb="tab"] {
    color: var(--uom-text) !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--uom-blue) !important;
    border-bottom-color: var(--uom-blue) !important;
}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: var(--uom-blue) !important;
    border-color: var(--uom-blue) !important;
    color: var(--uom-white) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--uom-blue-light) !important;
    border-color: var(--uom-blue-light) !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    border-color: var(--uom-blue) !important;
    color: var(--uom-blue) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--uom-blue-pale) !important;
    border-color: var(--uom-blue) !important;
    color: var(--uom-blue) !important;
}

/* Download button */
.stDownloadButton > button {
    border-color: var(--uom-blue) !important;
    color: var(--uom-blue) !important;
}
.stDownloadButton > button:hover {
    background-color: var(--uom-blue-pale) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--uom-blue) !important;
    font-weight: 600 !important;
}

/* Expander headers */
[data-testid="stExpander"] summary span {
    font-weight: 500 !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: var(--uom-blue) !important;
}

/* Selectbox & multiselect focus ring */
[data-baseweb="select"] [data-baseweb="tag"] {
    background-color: var(--uom-blue-pale) !important;
    color: var(--uom-blue) !important;
}

/* Success messages — keep green but tint slightly */
.stSuccess {
    border-left-color: var(--uom-blue) !important;
}

/* Sidebar (if ever used) */
section[data-testid="stSidebar"] {
    background-color: var(--uom-blue) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--uom-white) !important;
}

/* Divider under header banner */
.uom-banner {
    background: linear-gradient(135deg, #094183 0%, #0E6EB8 100%);
    padding: 1rem 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.uom-banner h2 {
    color: white !important;
    margin: 0 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
.uom-banner p {
    color: rgba(255, 255, 255, 0.85) !important;
    margin: 0.25rem 0 0 0 !important;
    font-size: 0.95rem !important;
}
</style>
"""
st.markdown(_UOM_FONT, unsafe_allow_html=True)
st.markdown(_UOM_CSS, unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────

def _check(result: dict | None):
    if result is None:
        return None
    if result.get("error"):
        st.error(result.get("message", "Unknown error"))
        return None
    return result


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


# ── Session state defaults ─────────────────────────────────────────

DEFAULTS = {
    "api_key": "",
    "api_key_valid": False,
    "models": [],
    "selected_model": "",
    "thinking_effort": "high",
    "config": None,
    "selected_years": [],
    "selected_countries": [],
    "selected_section": "",
    "fallback_section": "",
    "filtered_count": None,
    "task_prompt": "",
    "rule_count": 0,
    "show_rules": False,
    "taxonomy_count": 0,
    "cost_estimate": None,
    "trial_result": None,
    "trial_confirmed": False,
    "saved_prompt": None,
    "trial_store": {},
    "jev_api_key": "",
    "jev_api_key_valid": False,
    "api_key_input_n": 0,
    "jev_key_input_n": 0,
    "batch_run_token": None,
    "batch_snapshot": None,
    "batch_limits": {},
    "batch_last_phase": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Load config on first run ───────────────────────────────────────

# Shared by all sessions: on Turso this takes about 30 s.
@st.cache_data(show_spinner="Loading the database...")
def _cached_filter_options():
    return database.get_filter_options()


if st.session_state.config is None:
    try:
        cfg = _cached_filter_options()
        st.session_state.config = cfg
    except Exception:
        # Visitors only see the message below. The details go to the server log
        # (the terminal locally, "Manage app" on Streamlit Cloud).
        traceback.print_exc()

st.markdown(
    '<div class="uom-banner">'
    "<h2>ASX/NZX Annual Reports Data Extraction App</h2>"
    "<p>Extract and classify information from ASX/NZX annual reports using AI</p>"
    "</div>",
    unsafe_allow_html=True,
)

config = st.session_state.config
if config is None:
    st.error(
        "Could not load configuration. Please try again later. "
        "If you run this app, the details are in the server log."
    )
    st.stop()


# ── Helper: build filter params dict ───────────────────────────────

def _filter_params() -> dict:
    return {
        "years": st.session_state.selected_years,
        "countries": st.session_state.selected_countries,
        "section_name": st.session_state.selected_section,
        "fallback_section": st.session_state.fallback_section,
    }


def _get_rules() -> list[str]:
    if not st.session_state.get("show_rules"):
        return []
    return [
        st.session_state.get(f"rule_{i}", "").strip()
        for i in range(st.session_state.get("rule_count", 0))
        if st.session_state.get(f"rule_{i}", "").strip()
    ]


def _get_taxonomy() -> list[dict]:
    taxonomy = []
    for i in range(st.session_state.get("taxonomy_count", 0)):
        code = st.session_state.get(f"tax_{i}_code", "").strip()
        if not code:
            continue
        entry = {
            "code": code,
            "definition": st.session_state.get(f"tax_{i}_definition", "").strip(),
        }
        if st.session_state.get(f"tax_{i}_show_cats"):
            cats = []
            for j in range(st.session_state.get(f"tax_{i}_cat_count", 0)):
                name = st.session_state.get(f"tax_{i}_cat_{j}_name", "").strip()
                if name:
                    cats.append({
                        "name": name,
                        "description": st.session_state.get(f"tax_{i}_cat_{j}_desc", "").strip(),
                    })
            if cats:
                entry["categories"] = cats
        taxonomy.append(entry)
    return taxonomy


def _is_valid_code(code: str) -> bool:
    if not code or not code[0].isalpha():
        return False
    return all(c.isalnum() or c == "_" for c in code)


def _clear_taxonomy_state():
    for i in range(st.session_state.get("taxonomy_count", 0)):
        for j in range(st.session_state.get(f"tax_{i}_cat_count", 0)):
            for suffix in ("_name", "_desc"):
                key = f"tax_{i}_cat_{j}{suffix}"
                if key in st.session_state:
                    del st.session_state[key]
        for suffix in ("_code", "_definition", "_show_cats", "_cat_count"):
            key = f"tax_{i}{suffix}"
            if key in st.session_state:
                del st.session_state[key]
    st.session_state.taxonomy_count = 0


def _delete_taxonomy_entry(idx: int):
    count = st.session_state.taxonomy_count
    for j in range(st.session_state.get(f"tax_{idx}_cat_count", 0)):
        for suffix in ("_name", "_desc"):
            key = f"tax_{idx}_cat_{j}{suffix}"
            if key in st.session_state:
                del st.session_state[key]
    for i in range(idx, count - 1):
        nxt = i + 1
        for suffix in ("_code", "_definition"):
            st.session_state[f"tax_{i}{suffix}"] = st.session_state.get(f"tax_{nxt}{suffix}", "")
        st.session_state[f"tax_{i}_show_cats"] = st.session_state.get(f"tax_{nxt}_show_cats", False)
        nxt_cat_count = st.session_state.get(f"tax_{nxt}_cat_count", 0)
        st.session_state[f"tax_{i}_cat_count"] = nxt_cat_count
        for j in range(nxt_cat_count):
            for suffix in ("_name", "_desc"):
                st.session_state[f"tax_{i}_cat_{j}{suffix}"] = st.session_state.get(f"tax_{nxt}_cat_{j}{suffix}", "")
    last = count - 1
    for suffix in ("_code", "_definition", "_show_cats", "_cat_count"):
        key = f"tax_{last}{suffix}"
        if key in st.session_state:
            del st.session_state[key]
    for j in range(30):
        for suffix in ("_name", "_desc"):
            key = f"tax_{last}_cat_{j}{suffix}"
            if key in st.session_state:
                del st.session_state[key]
    st.session_state.taxonomy_count = count - 1


def _delete_category(tax_idx: int, cat_idx: int):
    cat_count = st.session_state.get(f"tax_{tax_idx}_cat_count", 0)
    for j in range(cat_idx, cat_count - 1):
        for suffix in ("_name", "_desc"):
            st.session_state[f"tax_{tax_idx}_cat_{j}{suffix}"] = st.session_state.get(
                f"tax_{tax_idx}_cat_{j + 1}{suffix}", ""
            )
    last = cat_count - 1
    for suffix in ("_name", "_desc"):
        key = f"tax_{tax_idx}_cat_{last}{suffix}"
        if key in st.session_state:
            del st.session_state[key]
    st.session_state[f"tax_{tax_idx}_cat_count"] = cat_count - 1
    if cat_count - 1 == 0:
        st.session_state[f"tax_{tax_idx}_show_cats"] = False


def _classify_params() -> dict:
    return {
        **_filter_params(),
        "model": st.session_state.selected_model,
        "task": st.session_state.task_prompt,
        "rules": _get_rules(),
        "taxonomy": _get_taxonomy(),
        "thinking_effort": st.session_state.thinking_effort,
    }


is_ready = (
    st.session_state.api_key_valid
    and st.session_state.filtered_count
    and st.session_state.filtered_count["count"] > 0
    and st.session_state.task_prompt.strip()
    and len(_get_taxonomy()) > 0
)


@st.dialog("Confirm Trial Run", width="large")
def _confirm_trial_dialog():
    fp = _filter_params()
    cp = _classify_params()

    st.markdown("**Are you sure you want to run the trial?**")
    st.markdown("The following configuration will be used:")

    c1, c2 = st.columns(2)
    c1.markdown(f"**Years:** {', '.join(str(y) for y in fp['years'])}")
    c2.markdown(f"**Countries:** {', '.join(fp['countries'])}")
    c1.markdown(f"**Primary Section:** {fp['section_name']}")
    c2.markdown(f"**Fallback Section:** {fp.get('fallback_section') or '—'}")

    st.divider()
    st.markdown("**User Message Template**")
    preview = build_user_message(
        company="{company}",
        year="{year}",
        section=fp["section_name"],
        content="{document content}",
        task=cp["task"],
        rules=cp["rules"],
        taxonomy=cp["taxonomy"],
        section_description=database.SECTION_DESCRIPTIONS.get(fp["section_name"]),
    )
    st.code(preview, language=None)

    col_yes, col_no = st.columns(2)
    if col_yes.button("Yes, run trial", type="primary", use_container_width=True):
        st.session_state.trial_confirmed = True
        st.rerun()
    if col_no.button("Cancel", use_container_width=True):
        st.session_state.trial_confirmed = False
        st.rerun()


def _run_jev_if_enabled(result: dict, raw_samples: list[dict], cp: dict):
    if not st.session_state.jev_api_key_valid:
        return
    taxonomy = cp.get("taxonomy") or []
    taxonomy_codes = [e["code"] for e in taxonomy]
    rules = cp.get("rules") or []
    if not taxonomy_codes:
        return
    anthropic_key = st.session_state.get("api_key")
    with st.spinner("Running Jev verification..."):
        verifications = _run_async(verify_trial_results(
            raw_samples,
            result["samples"],
            rules,
            taxonomy_codes,
            api_key=st.session_state.jev_api_key,
            anthropic_api_key=anthropic_key,
        ))
    for i, v in enumerate(verifications):
        if v is not None:
            result["samples"][i]["confidence"] = v


# ── Tabs ───────────────────────────────────────────────────────────

tab_bg, tab_setup, tab_filter, tab_prompt, tab_trial, tab_cost, tab_full = st.tabs([
    "Description",
    "(1) Setup",
    "(2) Filter Records",
    "(3) Extraction Prompt",
    "(4) Trial Run",
    "(5) Cost Estimate",
    "(6) Full Run",
])


# ── Background ───────────────────────────────────────────────────

with tab_bg:
    st.subheader("What does this AI app do?")
    st.markdown(
        "This tool uses **Claude** from Anthropic to extract and classify information from "
        "Australian (ASX) and New Zealand (NZX) company annual reports at scale."
        "Extract information from over thousands of files within hours."
    )
    st.markdown(
        "The database contains reports from over **~2,150 companies** "
        "covering financial years 2023–2025. Each report is split into standardised "
        "sections (e.g. Financial Statements, Directors Report, Remuneration Report) "
        "so you can target exactly the content you need."
    )

    st.subheader("How it works")
    st.markdown(
        "1. **Setup** — Enter your Claude API key and choose a model.\n"
        "2. **Filter** — Select which years, countries, and report sections to process.\n"
        "3. **Prompt** — Write an extraction prompt describing what data to pull from each report "
        "(e.g. classify industry, extract revenue figures, identify risk factors).\n"
        "4. **Cost Estimate** — See the estimated token usage and cost before committing.\n"
        "5. **Trial Run** — Test your prompt on 15 random samples to check quality.\n"
        "6. **Full Run** — Classify every filtered record, download the results as CSV, "
        "and resume unfinished runs by uploading the CSV again."
    )


# ── Tab 1: Setup ──────────────────────────────────────────────────

with tab_setup:
    st.subheader("Large Language Model: Claude (Mandatory)")
    col_key, col_model = st.columns(2)

    with col_key:
        # After validation the box is emptied (new key) so the key isn't left in a
        # password field — Firefox offers to save it when ?run= changes the URL.
        api_key = st.text_input(
            "Claude API Key", type="password",
            key=f"api_key_input_{st.session_state.api_key_input_n}",
            placeholder="Validated key in use — paste a new key to replace it"
            if st.session_state.api_key_valid else None,
            help="Paste your key and press Enter, then click Validate Key.",
        )
        st.caption("Press **Enter** after pasting your key, then click the button below.")
        if st.button("Validate Key"):
            if not api_key.strip():
                st.warning("Please enter an API key.")
            else:
                st.session_state.api_key = api_key.strip()
                with st.spinner("Validating API key..."):
                    val_result = classifier.validate_api_key(st.session_state.api_key)
                if val_result and val_result.get("valid"):
                    st.session_state.api_key_valid = True
                    st.session_state.models = val_result.get("models", [])
                    if st.session_state.models and not st.session_state.selected_model:
                        st.session_state.selected_model = st.session_state.models[0]["id"]
                    st.session_state.api_key_input_n += 1
                    st.success("API key validated successfully.")
                    st.rerun()
                else:
                    st.error(val_result.get("message", "Invalid API key.") if val_result else "Validation failed.")

        if st.session_state.api_key_valid:
            st.success("API key is valid.")
        st.caption("[How to get an API key](https://platform.claude.com/docs/en/get-api-key)")

    with col_model:
        model_options = st.session_state.models
        if not model_options:
            st.selectbox("Model", ["—"], disabled=True)
            st.caption("Validate your API key to load available models.")
        else:
            model_labels = [m["label"] for m in model_options]
            model_ids = [m["id"] for m in model_options]

            current_idx = 0
            if st.session_state.selected_model in model_ids:
                current_idx = model_ids.index(st.session_state.selected_model)

            selected_label = st.selectbox("Model", model_labels, index=current_idx)
            selected_idx = model_labels.index(selected_label)
            st.session_state.selected_model = model_ids[selected_idx]

            effort_options = ["low", "medium", "high", "xhigh", "max"]
            effort_labels = ["Low", "Medium", "High (default)", "XHigh", "Max"]
            current_effort = st.session_state.thinking_effort
            effort_idx = effort_options.index(current_effort) if current_effort in effort_options else 2
            selected_effort = st.selectbox(
                "Thinking Effort",
                effort_options,
                index=effort_idx,
                format_func=lambda e: effort_labels[effort_options.index(e)],
                help="Controls thinking depth. Higher effort = deeper reasoning but more tokens and cost.",
            )
            st.session_state.thinking_effort = selected_effort
        st.caption(
            "[Models overview](https://platform.claude.com/docs/en/models/overview) — Compare capabilities and context windows"
            " · [Pricing](https://platform.claude.com/docs/en/about-claude/pricing) — Token rates per model"
            " · [Thinking](https://academy.claude.com/courses/claude-platform-101/what-is-thinking) — How thinking effort affects reasoning"
        )

    st.divider()
    st.subheader("Jev Verification (Optional)")
    st.caption(
        "Add a TypeSafe API key to enable Jev confidence scoring on extraction results. "
        "Without this key, extractions still run — confidence scores are simply omitted."
    )

    col_jev_key, col_jev_status = st.columns(2)

    with col_jev_key:
        jev_key = st.text_input(
            "TypeSafe API Key",
            type="password",
            key=f"jev_key_input_{st.session_state.jev_key_input_n}",
            placeholder="Validated key in use — paste a new key to replace it"
            if st.session_state.jev_api_key_valid else None,
            help="Paste your TypeSafe API key and press Enter, then click Validate.",
        )
        st.caption("Press **Enter** after pasting your key, then click the button below.")
        if st.button("Validate Jev Key"):
            if not jev_key.strip():
                st.warning("Please enter a TypeSafe API key.")
            else:
                st.session_state.jev_api_key = jev_key.strip()
                with st.spinner("Validating TypeSafe API key..."):
                    valid = validate_jev_api_key(st.session_state.jev_api_key)
                if valid:
                    st.session_state.jev_api_key_valid = True
                    st.session_state.jev_key_input_n += 1
                    st.success("TypeSafe API key validated.")
                    st.rerun()
                else:
                    st.session_state.jev_api_key_valid = False
                    st.error("Invalid TypeSafe API key.")

        if st.session_state.jev_api_key_valid:
            st.success("TypeSafe API key is valid.")
        st.caption("[Get a TypeSafe API key](https://console.typesafe.ai/login?returnTo=%2Fhome)")

    with col_jev_status:
        if st.session_state.jev_api_key_valid:
            st.info("Jev verification **enabled** — confidence scores will be computed after each extraction.")
        else:
            st.info("Jev verification **disabled** — extractions will run without confidence scoring.")


# ── Tab 2: Filter Records ─────────────────────────────────────────

with tab_filter:
    col_years, col_countries = st.columns(2)

    _YEAR_ALL = "All"
    _COUNTRY_ALL = "All (ANZ)"

    with col_years:
        year_options = [_YEAR_ALL] + list(config["years"])
        if "year_ms" not in st.session_state:
            st.session_state["year_ms"] = list(st.session_state.selected_years)

        def _expand_all_years():
            if _YEAR_ALL in st.session_state["year_ms"]:
                st.session_state["year_ms"] = list(config["years"])

        st.multiselect(
            "Years", year_options, key="year_ms",
            on_change=_expand_all_years,
            format_func=lambda y: "All Years" if y == _YEAR_ALL else str(y),
        )
        st.session_state.selected_years = [
            y for y in st.session_state["year_ms"] if y != _YEAR_ALL
        ]

    with col_countries:
        country_labels = {"AUS": "Australia", "NZ": "New Zealand"}
        country_options = [_COUNTRY_ALL] + list(config["countries"])
        if "country_ms" not in st.session_state:
            st.session_state["country_ms"] = list(st.session_state.selected_countries)

        def _expand_all_countries():
            if _COUNTRY_ALL in st.session_state["country_ms"]:
                st.session_state["country_ms"] = list(config["countries"])

        st.multiselect(
            "Countries", country_options, key="country_ms",
            on_change=_expand_all_countries,
            format_func=lambda c: "All (ANZ)" if c == _COUNTRY_ALL else country_labels.get(c, c),
        )
        st.session_state.selected_countries = [
            c for c in st.session_state["country_ms"] if c != _COUNTRY_ALL
        ]

    with st.expander("Section descriptions"):
        for name, desc in database.SECTION_DESCRIPTIONS.items():
            st.markdown(f"**{name}** — {desc}")

    col_section, col_fallback = st.columns(2)

    with col_section:
        sections = config["sections"]
        section_idx = 0
        if st.session_state.selected_section in sections:
            section_idx = sections.index(st.session_state.selected_section)
        st.session_state.selected_section = st.selectbox(
            "Primary Section",
            sections,
            index=section_idx,
            help=database.SECTION_DESCRIPTIONS.get(sections[section_idx] if sections else "", ""),
        )
        st.caption("The primary section is used first for each company. All matching records are included.")

    with col_fallback:
        fallback_options = ["None"] + [s for s in sections if s != st.session_state.selected_section]
        fb_idx = 0
        if st.session_state.fallback_section in fallback_options:
            fb_idx = fallback_options.index(st.session_state.fallback_section)
        fb_selection = st.selectbox("Fallback Section", fallback_options, index=fb_idx)
        st.session_state.fallback_section = fb_selection if fb_selection != "None" else None
        st.caption("Only used for companies where the primary section is not available. Fills gaps in coverage.")

    st.caption("Please confirm selection of the records to be used in the information extraction or classification.")
    if st.button("Confirm Selection"):
        if not st.session_state.selected_years or not st.session_state.selected_countries:
            st.warning("Please select at least one year and one country.")
        else:
            fp = _filter_params()
            with st.spinner("Counting records..."):
                result = database.get_filtered_count(
                    fp["years"], fp["countries"], fp["section_name"], fp.get("fallback_section"),
                )
            if result:
                st.session_state.filtered_count = result

    if st.session_state.filtered_count:
        fc = st.session_state.filtered_count
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{fc['count']:,}")
        c2.metric("Primary Section", f"{fc['primary_count']:,}")
        c3.metric("Fallback Section", f"{fc['fallback_count']:,}")


# ── Tab 3: Prompt Editor ──────────────────────────────────────────

with tab_prompt:
    st.markdown(
        "Build your extraction prompt by filling in the sections below. "
        "The model automatically receives the document context (company, year, section) "
        "along with your task and rules."
    )

    EXAMPLE_TASK_INDUSTRY = (
        "Classify the industry of this company based on the **Principal Activities** "
        "subsection of the document only. Ignore all other content in the document "
        "(e.g. financial results, outlook, director biographies, remuneration)."
    )

    EXAMPLE_RULES_INDUSTRY = [
        "Classify based on the company's principal business activity — not its investments, customers, ownership structure, or secondary activities.",
        "If the company operates across multiple industries, choose the one that best represents its primary activity.",
    ]

    EXAMPLE_TAXONOMY_INDUSTRY = [
        {
            "code": "industry",
            "definition": "The industry classification based on the company's primary business activity as described in the Principal Activities section.",
            "categories": [
                {"name": "Mining", "description": "Exploration, extraction, processing or production of minerals, metals, coal, and other mined natural resources. Includes mining services where the company's primary activity is supporting mineral extraction."},
                {"name": "Energy", "description": "Exploration, production, generation, transmission, distribution or retailing of energy, including oil, gas, electricity and renewable energy. Includes companies primarily engaged in energy production or supply rather than mining fuels."},
                {"name": "Agriculture & Food", "description": "Cultivation, breeding, harvesting or production of agricultural commodities, food products and beverages. Includes farming, fishing, forestry, food processing and beverage manufacturing."},
                {"name": "Manufacturing", "description": "Design, manufacture, assembly or processing of physical goods not classified elsewhere. Includes industrial products, chemicals, packaging, machinery, consumer products and building materials."},
                {"name": "Construction & Engineering", "description": "Construction, engineering, contracting and infrastructure project delivery. Includes residential, commercial and civil construction, engineering consulting and specialist contracting services."},
                {"name": "Transport & Logistics", "description": "Transportation of passengers or freight and the provision of logistics, warehousing, distribution, shipping, aviation, rail, courier and supply chain services."},
                {"name": "Retail", "description": "Sale of goods directly to consumers through physical stores, online platforms or other retail channels. Includes retailers whose primary activity is merchandising rather than manufacturing."},
                {"name": "Consumer & Business Services", "description": "Provision of services directly to consumers, including hospitality, tourism, leisure, entertainment, education, personal services and gaming. Can also include services to business such as consulting, employment agencies and security firms. Excludes healthcare and financial services."},
                {"name": "Healthcare & Life Sciences", "description": "Provision of healthcare services, medical devices, pharmaceuticals, biotechnology, pathology, aged care and life sciences research. Includes companies primarily engaged in improving human or animal health."},
                {"name": "Banking", "description": "Acceptance of deposits and provision of banking services, including retail, commercial and investment banking. Includes authorised deposit-taking institutions and other licensed banks."},
                {"name": "Financial Services", "description": "Provision of financial products or services other than banking. Includes insurance, wealth management, investment management, superannuation, financial advice, stockbroking, exchanges, payment services and non-bank lending."},
                {"name": "Property & Real Estate", "description": "Ownership, development, investment, management or operation of real estate assets. Includes property developers, REITs, commercial and residential property owners, and real estate management businesses."},
                {"name": "Technology & Digital Services", "description": "Development or provision of software, information technology, digital platforms, cloud computing, cybersecurity, artificial intelligence, data analytics and other technology-based services. Includes IT consulting and software-as-a-service providers."},
                {"name": "Telecommunications & Media", "description": "Provision of telecommunications networks and communication services, together with media production, broadcasting, publishing, advertising and digital content businesses."},
                {"name": "Utilities & Infrastructure", "description": "Ownership or operation of essential public infrastructure and utility assets, including water, electricity and gas networks, waste management, airports, ports, toll roads and other regulated infrastructure."},
            ],
        },
    ]

    EXAMPLE_TASK_REVENUE = (
        "1. Extract the primary revenue line item from the consolidated statement of "
        "profit or loss (also known as the income statement, statement of comprehensive "
        "income, or statement of financial performance).\n"
        "2. Extract the value for the **target year** only. "
        "If two year columns are shown, use the column that matches the target year, "
        "not the comparative prior year.\n"
        "3. Identify the rounding unit shown at the top of the statement columns."
    )

    EXAMPLE_RULES_REVENUE = [
        "The revenue line item MUST include the word \"Revenue\", \"Sales\", or \"Turnover\" (case-insensitive).",
        "Do NOT use line items such as \"Other Income\", \"Finance Income\", \"Interest Income\", \"Gain on sale\", \"Fair value gain\", \"Government grants\", \"Deferred Revenue\", or any other non-primary revenue item. It must be the primary revenue line item.",
        "Determine the rounding unit by looking at the top of the statement columns.",
        "Extract the value for the **target year** only — do not use the comparative prior year column.",
    ]

    EXAMPLE_TAXONOMY_REVENUE = [
        {
            "code": "revenue",
            "definition": "The primary revenue amount for the target year, exactly as it appears in the document.",
        },
        {
            "code": "roundingunit",
            "definition": "The rounding unit used in the primary financial statements, as shown at the top of the statement columns.",
            "categories": [
                {"name": "Full Dollars", "description": "Figures are presented in full dollar amounts with no rounding."},
                {"name": "$'000", "description": "Figures are presented in thousands of dollars."},
                {"name": "$M", "description": "Figures are presented in millions of dollars."},
            ],
        },
        {
            "code": "revenuereported",
            "definition": "Whether a primary line item containing the word \"Revenue\", \"Sales\", or \"Turnover\" (case-insensitive) is reported in the statement.",
            "categories": [
                {"name": "Yes", "description": "A primary line item containing the word \"Revenue\", \"Sales\", or \"Turnover\" is present."},
                {"name": "No", "description": "No such primary line item is present."},
            ],
        },
    ]

    def _set_example(task, rules, taxonomy=None):
        st.session_state.task_prompt = task
        for i in range(st.session_state.get("rule_count", 0)):
            if f"rule_{i}" in st.session_state:
                del st.session_state[f"rule_{i}"]
        for i, r in enumerate(rules):
            st.session_state[f"rule_{i}"] = r
        st.session_state.rule_count = len(rules)
        st.session_state.show_rules = len(rules) > 0
        _clear_taxonomy_state()
        if taxonomy:
            for i, entry in enumerate(taxonomy):
                st.session_state[f"tax_{i}_code"] = entry["code"]
                st.session_state[f"tax_{i}_definition"] = entry["definition"]
                if "categories" in entry:
                    st.session_state[f"tax_{i}_show_cats"] = True
                    st.session_state[f"tax_{i}_cat_count"] = len(entry["categories"])
                    for j, cat in enumerate(entry["categories"]):
                        st.session_state[f"tax_{i}_cat_{j}_name"] = cat["name"]
                        st.session_state[f"tax_{i}_cat_{j}_desc"] = cat["description"]
                else:
                    st.session_state[f"tax_{i}_show_cats"] = False
                    st.session_state[f"tax_{i}_cat_count"] = 0
            st.session_state.taxonomy_count = len(taxonomy)

    col_ex1, col_ex2, col_clear = st.columns([2, 2, 1])
    with col_ex1:
        if st.button("Example: Industry Classification"):
            _set_example(EXAMPLE_TASK_INDUSTRY, EXAMPLE_RULES_INDUSTRY, EXAMPLE_TAXONOMY_INDUSTRY)
            st.rerun()
        st.caption("Section: Business Review")
    with col_ex2:
        if st.button("Example: Revenue Extraction"):
            _set_example(EXAMPLE_TASK_REVENUE, EXAMPLE_RULES_REVENUE, EXAMPLE_TAXONOMY_REVENUE)
            st.rerun()
        st.caption("Section: Financial Statements")
    with col_clear:
        if st.button("Clear All"):
            _set_example("", [])
            st.rerun()

    st.session_state.task_prompt = st.text_area(
        "Task ✱",
        value=st.session_state.task_prompt,
        height=150,
        help=(
            "Describe what data to extract or how to classify each report section. "
            "Specify the exact JSON output format you expect. "
            "This instruction is sent to the model for every record."
        ),
        placeholder="e.g. Classify this company's primary industry based on the report content...",
    )

    if not st.session_state.task_prompt.strip():
        st.warning("A task is required before you can run a trial.")

    st.divider()

    if not st.session_state.show_rules:
        if st.button("+ Add Rules (optional)", help="Add constraints the model must follow"):
            st.session_state.show_rules = True
            if st.session_state.rule_count == 0:
                st.session_state.rule_count = 1
                st.session_state["rule_0"] = ""
            st.rerun()
    else:
        col_heading, col_close = st.columns([10, 1])
        with col_heading:
            st.markdown("**Rules** *(optional)*")
            st.caption(
                "Add specific constraints the model must follow — classification boundaries, "
                "tie-breaking logic, output restrictions, etc. Up to 10 rules."
            )
        with col_close:
            if st.button(":material/close:", key="close_rules", help="Remove all rules"):
                for i in range(st.session_state.rule_count):
                    if f"rule_{i}" in st.session_state:
                        del st.session_state[f"rule_{i}"]
                st.session_state.rule_count = 0
                st.session_state.show_rules = False
                st.rerun()

        for i in range(st.session_state.rule_count):
            col_rule, col_del = st.columns([20, 1])
            with col_rule:
                st.text_input(
                    f"Rule {i + 1}",
                    key=f"rule_{i}",
                    label_visibility="collapsed",
                    placeholder=f"Rule {i + 1}",
                )
            with col_del:
                if st.button(":material/close:", key=f"del_rule_{i}", help="Remove this rule"):
                    for j in range(i, st.session_state.rule_count - 1):
                        st.session_state[f"rule_{j}"] = st.session_state.get(f"rule_{j + 1}", "")
                    last_key = f"rule_{st.session_state.rule_count - 1}"
                    if last_key in st.session_state:
                        del st.session_state[last_key]
                    st.session_state.rule_count -= 1
                    if st.session_state.rule_count == 0:
                        st.session_state.show_rules = False
                    st.rerun()

        if st.session_state.rule_count < 10:
            if st.button("+ Add Rule"):
                st.session_state[f"rule_{st.session_state.rule_count}"] = ""
                st.session_state.rule_count += 1
                st.rerun()
        else:
            st.caption("Maximum of 10 rules reached.")

    st.divider()

    # ── Taxonomy section (mandatory) ──
    st.markdown("**Taxonomy** ✱ — Define the fields you want to extract")
    st.caption(
        "Each field becomes a property in the structured JSON output. "
        "**Code** is the JSON key name (e.g. `revenue`). "
        "**Definition** tells the model what to extract. "
        "Optionally, add **Categories** to constrain output to a fixed set of values."
    )

    for i in range(st.session_state.get("taxonomy_count", 0)):
        col_code, col_del = st.columns([6, 1])
        with col_code:
            st.text_input(
                f"Code {i + 1}",
                key=f"tax_{i}_code",
                label_visibility="collapsed",
                placeholder="Code (e.g. total_revenue)",
            )
        with col_del:
            if st.button(":material/close:", key=f"del_tax_{i}", help="Remove this field"):
                _delete_taxonomy_entry(i)
                st.rerun()

        code_val = st.session_state.get(f"tax_{i}_code", "")
        if code_val and not _is_valid_code(code_val):
            st.error("Code must start with a letter and contain only letters, numbers, and underscores.")

        st.text_input(
            f"Definition {i + 1}",
            key=f"tax_{i}_definition",
            label_visibility="collapsed",
            placeholder="Definition: what should the model extract for this field?",
        )

        if not st.session_state.get(f"tax_{i}_show_cats"):
            if st.button("+ Add Categories", key=f"add_cats_{i}", help="Add predefined category options"):
                st.session_state[f"tax_{i}_show_cats"] = True
                if st.session_state.get(f"tax_{i}_cat_count", 0) == 0:
                    st.session_state[f"tax_{i}_cat_count"] = 1
                    st.session_state[f"tax_{i}_cat_0_name"] = ""
                    st.session_state[f"tax_{i}_cat_0_desc"] = ""
                st.rerun()
        else:
            with st.container(border=True):
                col_cat_heading, col_cat_close = st.columns([10, 1])
                with col_cat_heading:
                    st.markdown("**Categories**")
                with col_cat_close:
                    if st.button(":material/close:", key=f"close_cats_{i}", help="Remove all categories"):
                        for j in range(st.session_state.get(f"tax_{i}_cat_count", 0)):
                            for suffix in ("_name", "_desc"):
                                key = f"tax_{i}_cat_{j}{suffix}"
                                if key in st.session_state:
                                    del st.session_state[key]
                        st.session_state[f"tax_{i}_cat_count"] = 0
                        st.session_state[f"tax_{i}_show_cats"] = False
                        st.rerun()

                st.info(
                    "Categories constrain the model to output **exactly one** of the listed values "
                    "— it cannot select multiple or write free text when categories are defined. "
                    "A \"Don't Know\" fallback option is automatically included. "
                    "If you don't add categories, the field accepts free-text responses.",
                    icon=":material/info:",
                )

                cat_count = st.session_state.get(f"tax_{i}_cat_count", 0)
                for j in range(cat_count):
                    col_name, col_desc, col_cdel = st.columns([3, 7, 1])
                    with col_name:
                        st.text_input(
                            f"Cat name {i}_{j}",
                            key=f"tax_{i}_cat_{j}_name",
                            label_visibility="collapsed",
                            placeholder="Category name",
                        )
                    with col_desc:
                        st.text_input(
                            f"Cat desc {i}_{j}",
                            key=f"tax_{i}_cat_{j}_desc",
                            label_visibility="collapsed",
                            placeholder="Category description",
                        )
                    with col_cdel:
                        if st.button(":material/close:", key=f"del_cat_{i}_{j}", help="Remove this category"):
                            _delete_category(i, j)
                            st.rerun()

                st.caption("A \"Don't Know\" option is automatically included in the output schema.")

                if cat_count < 30:
                    if st.button("+ Add Category", key=f"add_cat_{i}"):
                        st.session_state[f"tax_{i}_cat_{cat_count}_name"] = ""
                        st.session_state[f"tax_{i}_cat_{cat_count}_desc"] = ""
                        st.session_state[f"tax_{i}_cat_count"] = cat_count + 1
                        st.rerun()
                else:
                    st.caption("Maximum of 30 categories reached.")

        if i < st.session_state.get("taxonomy_count", 0) - 1:
            st.markdown("---")

    if st.session_state.get("taxonomy_count", 0) < 20:
        if st.button("+ Add Field"):
            idx = st.session_state.get("taxonomy_count", 0)
            st.session_state[f"tax_{idx}_code"] = ""
            st.session_state[f"tax_{idx}_definition"] = ""
            st.session_state.taxonomy_count = idx + 1
            st.rerun()
    else:
        st.caption("Maximum of 20 fields reached.")

    if st.session_state.get("taxonomy_count", 0) == 0:
        st.warning("At least one taxonomy field is required.")

    st.divider()

    taxonomy = _get_taxonomy()
    can_save = bool(st.session_state.task_prompt.strip()) and len(taxonomy) > 0
    if st.button("Save Prompt", type="primary", disabled=not can_save):
        st.session_state.saved_prompt = {
            "task": st.session_state.task_prompt,
            "rules": _get_rules(),
            "taxonomy": taxonomy,
        }
        st.success("Prompt saved.")

    if st.session_state.saved_prompt:
        saved = st.session_state.saved_prompt
        rule_count = len(saved["rules"])
        rule_info = f" with {rule_count} rule{'s' if rule_count != 1 else ''}" if rule_count else ""
        tax_count = len(saved.get("taxonomy", []))
        tax_info = f", {tax_count} field{'s' if tax_count != 1 else ''}" if tax_count else ""
        st.caption(f"Saved prompt: task ({len(saved['task'])} chars){rule_info}{tax_info}")


# ── Tab 4: Trial Run ──────────────────────────────────────────────

with tab_trial:
    if not is_ready:
        st.info("Complete Setup, Filter, and Prompt tabs first.")
    else:
        saved = st.session_state.saved_prompt
        prompt_saved = saved == {
            "task": st.session_state.task_prompt,
            "rules": _get_rules(),
            "taxonomy": _get_taxonomy(),
        }
        if not prompt_saved:
            if saved:
                st.warning(
                    "Your prompt has changed since it was saved. Click **Save Prompt** at the "
                    "bottom of **(3) Extraction Prompt** to run a trial with the new prompt."
                )
            else:
                st.warning(
                    "Click **Save Prompt** at the bottom of **(3) Extraction Prompt** "
                    "before running a trial."
                )
        run_trial_clicked = st.button("Run Trial", disabled=not prompt_saved)
        st.caption(
            "Runs your extraction prompt against 15 randomly sampled records "
            "from the filtered set to preview results and estimate full-run cost."
        )

        if run_trial_clicked:
            st.session_state.trial_confirmed = False
            _confirm_trial_dialog()

        if st.session_state.get("trial_confirmed"):
            st.session_state.trial_confirmed = False
            fp = _filter_params()
            cp = _classify_params()
            sample_result = database.get_sample_records(
                fp["years"], fp["countries"], fp["section_name"], fp.get("fallback_section"), n=15,
            )
            samples = sample_result["samples"]
            if not samples:
                st.error("No records match the current filters.")
            else:
                with st.spinner("Running trial on 15 random samples... This may take 30-60 seconds."):
                    result = _run_async(classifier.run_trial(
                        samples, cp["model"], cp["task"], cp["rules"],
                        taxonomy=cp["taxonomy"],
                        thinking_effort=cp["thinking_effort"],
                        api_key=st.session_state.api_key,
                        trial_store=st.session_state.trial_store,
                    ))
                result = _check(result)
                if result:
                    _run_jev_if_enabled(result, samples, cp)
                    st.session_state.trial_result = result

                    successful = [s for s in result["samples"] if s["status"] == "success"]
                    if successful:
                        avg_in = sum(s["input_tokens"] for s in successful) / len(successful)
                        avg_out = sum(s["output_tokens"] for s in successful) / len(successful)
                        record_count = st.session_state.filtered_count["count"]
                        model_used = successful[0].get("model_used", cp["model"])
                        prices = match_pricing(model_used)
                        total_in = avg_in * record_count
                        total_out = avg_out * record_count
                        cost_usd = total_in / 1_000_000 * prices["input"] + total_out / 1_000_000 * prices["output"]
                        st.session_state.cost_estimate = {
                            "record_count": record_count,
                            "avg_input_tokens": int(avg_in),
                            "avg_output_tokens": int(avg_out),
                            "total_input_tokens": int(total_in),
                            "total_output_tokens": int(total_out),
                            "model": model_used,
                            "model_label": prices["label"],
                            "input_rate": prices["input"],
                            "output_rate": prices["output"],
                            "estimated_cost_usd": round(cost_usd, 2),
                            "trial_sample_count": len(successful),
                        }

        if st.session_state.trial_result:
            tr = st.session_state.trial_result
            c1, c2, c3 = st.columns(3)
            c1.metric("Succeeded", tr["success_count"])
            c2.metric("Errors", tr["error_count"])
            c3.metric("Trial Cost", f"A${tr['total_cost_usd'] * USD_TO_AUD:.4f}")

            col_retry, col_new = st.columns(2)
            with col_retry:
                if st.button("Retry Same", use_container_width=True):
                    trial_data = st.session_state.trial_store.get(tr["trial_id"])
                    if trial_data:
                        raw_samples = trial_data["raw_samples"]
                        cp = _classify_params()
                        with st.spinner("Re-running trial with same samples..."):
                            result = _run_async(classifier.run_trial(
                                raw_samples, cp["model"], cp["task"], cp["rules"],
                                taxonomy=cp["taxonomy"],
                                thinking_effort=cp["thinking_effort"],
                                api_key=st.session_state.api_key,
                                trial_store=st.session_state.trial_store,
                            ))
                        result = _check(result)
                        if result:
                            _run_jev_if_enabled(result, raw_samples, cp)
                            st.session_state.trial_result = result
                            successful = [s for s in result["samples"] if s["status"] == "success"]
                            if successful:
                                avg_in = sum(s["input_tokens"] for s in successful) / len(successful)
                                avg_out = sum(s["output_tokens"] for s in successful) / len(successful)
                                record_count = st.session_state.filtered_count["count"]
                                model_used = successful[0].get("model_used", cp["model"])
                                prices = match_pricing(model_used)
                                total_in = avg_in * record_count
                                total_out = avg_out * record_count
                                cost_usd = total_in / 1_000_000 * prices["input"] + total_out / 1_000_000 * prices["output"]
                                st.session_state.cost_estimate = {
                                    "record_count": record_count,
                                    "avg_input_tokens": int(avg_in),
                                    "avg_output_tokens": int(avg_out),
                                    "total_input_tokens": int(total_in),
                                    "total_output_tokens": int(total_out),
                                    "model": model_used,
                                    "model_label": prices["label"],
                                    "input_rate": prices["input"],
                                    "output_rate": prices["output"],
                                    "estimated_cost_usd": round(cost_usd, 2),
                                    "trial_sample_count": len(successful),
                                }
                            st.rerun()
            with col_new:
                if st.button("New Sample", use_container_width=True):
                    fp = _filter_params()
                    cp = _classify_params()
                    sample_result = database.get_sample_records(
                        fp["years"], fp["countries"], fp["section_name"],
                        fp.get("fallback_section"), n=15,
                    )
                    samples = sample_result["samples"]
                    if not samples:
                        st.error("No records match the current filters.")
                    else:
                        with st.spinner("Running trial on 15 new samples..."):
                            result = _run_async(classifier.run_trial(
                                samples, cp["model"], cp["task"], cp["rules"],
                                taxonomy=cp["taxonomy"],
                                thinking_effort=cp["thinking_effort"],
                                api_key=st.session_state.api_key,
                                trial_store=st.session_state.trial_store,
                            ))
                        result = _check(result)
                        if result:
                            _run_jev_if_enabled(result, samples, cp)
                            st.session_state.trial_result = result
                            successful = [s for s in result["samples"] if s["status"] == "success"]
                            if successful:
                                avg_in = sum(s["input_tokens"] for s in successful) / len(successful)
                                avg_out = sum(s["output_tokens"] for s in successful) / len(successful)
                                record_count = st.session_state.filtered_count["count"]
                                model_used = successful[0].get("model_used", cp["model"])
                                prices = match_pricing(model_used)
                                total_in = avg_in * record_count
                                total_out = avg_out * record_count
                                cost_usd = total_in / 1_000_000 * prices["input"] + total_out / 1_000_000 * prices["output"]
                                st.session_state.cost_estimate = {
                                    "record_count": record_count,
                                    "avg_input_tokens": int(avg_in),
                                    "avg_output_tokens": int(avg_out),
                                    "total_input_tokens": int(total_in),
                                    "total_output_tokens": int(total_out),
                                    "model": model_used,
                                    "model_label": prices["label"],
                                    "input_rate": prices["input"],
                                    "output_rate": prices["output"],
                                    "estimated_cost_usd": round(cost_usd, 2),
                                    "trial_sample_count": len(successful),
                                }
                            st.rerun()

            import pandas as pd

            with st.expander("What do Quotes, Supports and Confidence Scores mean?"):
                st.markdown(
                    "**Quotes:** verbatim excerpts from the report that Claude cites as "
                    "evidence for each value.\n\n"
                    "**Supports:** Claude's short argument for how the quotes lead to the "
                    "value it chose.\n\n"
                    "**Confidence scores** (only when Jev verification is enabled in "
                    "**(1) Setup**): TypeSafe's Jev runs a set of checks on each answer and "
                    "estimates the probability that each one failed, P(wrong). "
                    "A score is 1 − P(wrong), so 1.0 means confident and 0.0 means almost "
                    "certainly wrong.\n"
                    "- `quote_fabrication`: are any of the quotes missing from the document?\n"
                    "- `argument_support`: do the quotes and argument fail to support the value?\n"
                    "- `dont_know_validation`: for \"Don't Know\" answers, does the document "
                    "actually contain enough information to answer?\n"
                    "- **Rule Compliance**: did the result break one of your rules?\n\n"
                    "**The score in the table is the weakest link of the chain.** Each "
                    "confidence column shows the lowest score across all fields (or all rules) "
                    "in that record, so a single weak field pulls the whole column down. "
                    "For example, if `revenue` scores 0.95 on quote_fabrication and "
                    "`roundingunit` scores 0.60, `confidence_quote` shows 0.60.\n"
                    "- `confidence_quote` → quote_fabrication\n"
                    "- `confidence_reason` → argument_support\n"
                    "- `confidence_rule` → Rule Compliance\n"
                    "- `confidence_DN` → dont_know_validation (N/A if no field was \"Don't Know\")\n\n"
                    "Open a record under **Details** to see the individual checks for each field."
                )

            table_rows = []
            for sample in tr["samples"]:
                row = {
                    "Company": sample["company_name"],
                    "Year": sample["year"],
                    "Country": sample["country"],
                    "Section": sample["section_name"],
                    "Status": sample["status"],
                }
                if sample["status"] == "success" and isinstance(sample.get("result"), dict):
                    for key, value in sample["result"].items():
                        if key.endswith("_quote") or key.endswith("_supports") or key.endswith("_reasoning"):
                            continue
                        row[key] = value
                conf = sample.get("confidence")
                if conf and not conf.get("skipped"):
                    if conf.get("confidence_rule") is not None:
                        row["confidence_rule"] = conf["confidence_rule"]
                    if conf.get("confidence_quote") is not None:
                        row["confidence_quote"] = conf["confidence_quote"]
                    if conf.get("confidence_reason") is not None:
                        row["confidence_reason"] = conf["confidence_reason"]
                    dn = compute_dn_confidence(conf.get("field_confidences", {}))
                    # All strings: mixing "N/A" with floats breaks Arrow serialisation.
                    row["confidence_DN"] = "N/A" if dn is None else str(dn)
                table_rows.append(row)

            df = pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Details")
            for i, sample in enumerate(tr["samples"]):
                label = f"{sample['company_name']} ({sample['year']})"
                conf = sample.get("confidence")
                with st.expander(label):
                    if sample["status"] == "success" and isinstance(sample.get("result"), dict):
                        result_data = sample["result"]
                        verified = bool(conf) and not conf.get("skipped")

                        if conf and conf.get("skipped"):
                            reason = conf.get("skip_reason", "unknown")
                            tokens = conf.get("document_tokens_estimated", 0)
                            st.warning(
                                f"Verification skipped — {reason} "
                                f"(estimated {tokens:,} tokens)"
                            )
                        elif conf and conf.get("trimmed"):
                            kept = len(conf.get("sections_kept", []))
                            total = conf.get("sections_total", 0)
                            trimmed_tok = conf.get("document_tokens_trimmed", 0)
                            original_tok = conf.get("document_tokens_original", 0)
                            st.info(
                                f"Verified on {kept} of {total} document sections "
                                f"({trimmed_tok:,} / {original_tok:,} tokens)"
                            )

                        # One block per field: quotes, supports, then its Jev checks.
                        # Taxonomy order comes from the result keys (they follow the schema).
                        fc = conf["field_confidences"] if verified else {}
                        fields = [
                            k for k in result_data
                            if f"{k}_quote" in result_data or f"{k}_supports" in result_data
                        ]
                        fields += [k for k in fc if k not in fields]
                        for field in fields:
                            field_conf = fc.get(field)
                            heading = f"**{field}**"
                            lines = []
                            quote = result_data.get(f"{field}_quote", "")
                            supports = result_data.get(f"{field}_supports", "")
                            if quote:
                                lines.append(f"- **Quotes:** {quote}")
                            if supports:
                                lines.append(f"- **Supports:** {supports}")
                            for check in (field_conf or {}).get("checks", []):
                                lines.append(
                                    f"- `{check['check']}` P(wrong): {check['p_wrong']:.3f}"
                                )
                            st.markdown("\n\n".join([heading, "\n".join(lines)]))

                        if verified:
                            rule_checks = conf.get("rule_checks", [])
                            if rule_checks:
                                st.markdown("**Rule Compliance**")
                                for rc in rule_checks:
                                    st.markdown(
                                        f"- P(wrong): {rc['p_wrong']:.3f} — {rc['rule']}"
                                    )

                    thinking_text = sample.get("thinking_text", "")
                    if thinking_text:
                        st.markdown("**Claude Thinking**")
                        with st.container(height=300):
                            st.markdown(thinking_text)

                    source_text = sample.get("source_text", "")
                    if source_text:
                        word_count = len(source_text.split())
                        st.markdown("**Source Text**")
                        st.caption(f"{word_count:,} words")
                        with st.container(height=400):
                            st.markdown(source_text)
                    else:
                        st.info("Source text not available.")


# ── Tab 5: Cost Estimate ──────────────────────────────────────────

with tab_cost:
    if not st.session_state.cost_estimate:
        st.info("Run a trial first — cost is estimated from actual API usage.")
    else:
        ce = st.session_state.cost_estimate
        st.caption(
            f"Based on {ce['trial_sample_count']} successful trial samples, "
            "extrapolated to the full filtered record count."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{ce['record_count']:,}")
        c2.metric("Avg Input Tokens", f"{ce['avg_input_tokens']:,}")
        c3.metric("Avg Output Tokens", f"{ce['avg_output_tokens']:,}")
        c4.metric("Model", ce["model_label"])

        cost_aud = ce["estimated_cost_usd"] * USD_TO_AUD
        cost_low = cost_aud * 0.85
        cost_high = cost_aud * 1.15
        st.metric("Estimated Cost", f"A${cost_low:.2f} – A${cost_high:.2f}")


# ── Tab 6: Full Run ───────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_content_stats(years, countries, section_name, fallback_section):
    return database.get_content_stats(list(years), list(countries), section_name, fallback_section or None)


def _current_run():
    token = st.session_state.batch_run_token or st.query_params.get("run")
    run = batch_processor.get_run(token)
    if run is None:
        st.session_state.batch_run_token = None
        if "run" in st.query_params:
            del st.query_params["run"]
        return None, None
    st.session_state.batch_run_token = token
    if st.query_params.get("run") != token:
        st.query_params["run"] = token
    return token, run


def _attach_run(run):
    token = batch_processor.register_run(run)
    st.session_state.batch_run_token = token
    st.session_state.batch_snapshot = None
    st.session_state.batch_was_running = False
    st.query_params["run"] = token


def _limits_for(model: str) -> dict:
    if not st.session_state.api_key_valid:
        return start_tier_limits(model)
    key = f"{model}|{hash(st.session_state.api_key)}"
    cache = st.session_state.batch_limits
    if key not in cache:
        cache[key] = batch_processor.probe_rate_limits(st.session_state.api_key, model)
    return cache[key]


def _trial_matches(cfg) -> list[dict]:
    tr = st.session_state.trial_result
    if not tr:
        return []
    trial = st.session_state.trial_store.get(tr["trial_id"])
    if not trial:
        return []
    trial_cfg = RunConfig(
        model=trial["model"],
        thinking_effort=trial.get("thinking_effort", "high"),
        task=trial["task"],
        rules=tuple(trial.get("rules") or []),
        taxonomy=tuple(trial.get("taxonomy") or []),
    )
    if trial_cfg.config_id != cfg.config_id:
        return []
    return [s for s in tr["samples"] if s["status"] == "success"]


def _run_estimates(run) -> dict:
    stats = run.table.usage_stats()
    if stats and stats["n"] >= 10:
        return {**stats, "source": f"{stats['n']:,} completed rows of this run"}
    trial_ok = _trial_matches(run.config)
    if trial_ok:
        n = len(trial_ok)
        return {
            "n": n,
            "avg_input": sum(s["input_tokens"] for s in trial_ok) / n,
            "avg_output": sum(s["output_tokens"] for s in trial_ok) / n,
            "avg_latency_s": sum(s["latency_ms"] for s in trial_ok) / n / 1000,
            "source": f"your trial run ({n} samples)",
        }
    f = run.config.filters
    content = _cached_content_stats(
        tuple(f.get("years", [])), tuple(f.get("countries", [])),
        f.get("section_name", ""), f.get("fallback_section", ""),
    )
    est = estimate_cost(
        1, content["avg_length"], run.config.model, run.config.task,
        list(run.config.rules), list(run.config.taxonomy), run.config.thinking_effort,
    )
    return {
        "n": 0,
        "avg_input": est["avg_input_tokens_per_record"],
        "avg_output": est["estimated_output_tokens_per_record"],
        "avg_latency_s": 60.0,
        "source": "default assumptions (no trial with these settings)",
    }


def _fmt_minutes(m: float) -> str:
    if m < 1:
        return "< 1 min"
    if m < 90:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


LONG_RUN_WARN_MIN = 120


def _long_run_warning(full: dict, jev_on: bool):
    if full["minutes_high"] <= LONG_RUN_WARN_MIN:
        return
    jev_note = ", plus time for Jev verification" if jev_on else ""
    st.warning(
        f"**This run may take over {LONG_RUN_WARN_MIN / 60:g} hours**{jev_note}. "
        "If the app restarts during the run, every row you haven't downloaded is lost. "
        "We recommend splitting it into smaller runs in (2) Filter Records, for example one year at a time. "
        "If you go ahead, keep this page open, download the results about every 30 minutes, "
        "and download again as soon as it finishes."
    )


def _make_snapshot(run):
    st.session_state.batch_snapshot = {
        "data": run.table.to_csv_bytes(),
        "version": run.table.version,
        "token": st.session_state.batch_run_token,
    }


@st.dialog("Confirm Full Run", width="large")
def _confirm_full_run_dialog(n_left: int, jev_left: int, concurrency: int):
    _, run = _current_run()
    if run is None:
        st.error("Run not found.")
        return
    cfg = run.config
    est = _run_estimates(run)
    jev_on = st.session_state.jev_api_key_valid

    if n_left:
        full = estimate_full_run(
            n_left, est["avg_input"], est["avg_output"], cfg.model,
            est["avg_latency_s"], concurrency, run.limits,
        )
        st.markdown(
            f"**Classify {n_left:,} rows with {match_pricing(cfg.model)['label']} "
            f"({cfg.thinking_effort} effort)?**"
        )
        c1, c2 = st.columns(2)
        c1.metric("Estimated cost", f"A${full['cost_usd'] * USD_TO_AUD:,.2f}")
        c2.metric("Estimated time", f"{_fmt_minutes(full['minutes_low'])} – {_fmt_minutes(full['minutes_high'])}")
        st.caption(f"US${full['cost_usd']:,.2f}. Estimates from {est['source']}.")
        _long_run_warning(full, jev_on)
    else:
        st.markdown(f"**All rows are classified. Verify the remaining {jev_left:,} rows with Jev?**")
    if jev_on:
        st.markdown("Jev verification runs automatically after classification.")
    else:
        st.markdown("No Jev key set, so results won't be verified. Add a key in Setup before starting to verify them.")

    col_yes, col_no = st.columns(2)
    if col_yes.button("Yes, start", type="primary", use_container_width=True):
        batch_processor.start_classification(
            run,
            api_key=st.session_state.api_key,
            concurrency=concurrency,
            output_estimate=est["avg_output"],
            jev_api_key=st.session_state.jev_api_key if jev_on else None,
        )
        st.session_state.batch_was_running = True
        st.rerun()
    if col_no.button("Cancel", use_container_width=True):
        st.rerun()


def _render_progress(run):
    p = run.progress()
    if run.running:
        label = "Classifying" if p["phase"] == "classifying" else "Jev verification"
        frac = p["done"] / p["total"] if p["total"] else 1.0
        st.progress(min(frac, 1.0), text=f"{label}: {p['done']:,} / {p['total']:,}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Errors", f"{p['errors']:,}")
        c2.metric("Cost so far", f"A${run.table.counts()['cost_usd'] * USD_TO_AUD:,.2f}")
        c3.metric("Elapsed", _fmt_minutes(p["elapsed_s"] / 60))
        c4.metric("ETA", _fmt_minutes(p["eta_min"]) if p["eta_min"] is not None else "—")
        c5.metric("Docs/min", f"{p['docs_per_min']:.0f}")
        st.caption(f"{p['in_flight']} requests in flight")
        if p["last_error"]:
            st.caption(f"Last error: {p['last_error']}")
    elif p["phase"] == "failed":
        st.error(f"Run stopped: {p['stop_reason']}")
    elif p["phase"] == "stopped":
        st.warning("Run stopped. Press Resume to continue where it left off.")
    elif p["phase"] == "finished":
        st.success("Run finished. A fresh CSV snapshot is ready below.")

    if st.session_state.get("batch_was_running") and not run.running:
        st.session_state.batch_was_running = False
        _make_snapshot(run)
        st.rerun()


_render_progress_live = st.fragment(run_every=2)(_render_progress)


with tab_full:
    token, run = _current_run()

    if run is None:
        st.subheader("New run")
        checks = [
            ("Claude API key validated (Setup)", bool(st.session_state.api_key_valid)),
            ("Filters return records (Filter Records)", bool(
                st.session_state.filtered_count and st.session_state.filtered_count["count"] > 0
            )),
            ("Extraction prompt and taxonomy set (Extraction Prompt)", bool(
                st.session_state.task_prompt.strip() and _get_taxonomy()
            )),
            ("Trial run done (Trial Run)", st.session_state.trial_result is not None),
        ]
        for label, ok in checks:
            st.markdown(f"{'✅' if ok else '❌'} {label}")
        if st.button("Prepare run", type="primary", disabled=not all(ok for _, ok in checks)):
            fp = _filter_params()
            with st.spinner("Building the run table from your filters (can take ~30 s)..."):
                rows = database.get_filtered_rowids(
                    fp["years"], fp["countries"], fp["section_name"], fp.get("fallback_section") or None,
                )
                cfg = RunConfig.from_params(_classify_params())
                new_run = batch_processor.BatchRun(BatchTable.new(rows, cfg), cfg)
                new_run.limits = _limits_for(cfg.model)
            _attach_run(new_run)
            st.rerun()

        st.divider()
        st.subheader("Resume a run")
        st.caption(
            "Upload a results CSV from an earlier run. Only rows not yet 'success' will be processed. "
            "The model and Extraction Prompt settings must be the same as when the run was made."
        )
        resume_file = st.file_uploader("Results CSV", type=["csv"], key="resume_file")
        if st.button("Load run", disabled=resume_file is None):
            try:
                if not (st.session_state.task_prompt.strip() and _get_taxonomy() and st.session_state.selected_model):
                    raise ValueError(
                        "First set up the model and Extraction Prompt exactly as they were for this run."
                    )
                cfg = RunConfig.from_params(_classify_params())
                table = BatchTable.from_csv(resume_file.getvalue(), cfg)
            except ValueError as e:
                st.error(str(e))
            else:
                loaded = batch_processor.BatchRun(table, cfg)
                loaded.limits = _limits_for(cfg.model)
                _attach_run(loaded)
                st.rerun()

    else:
        cfg = run.config
        counts = run.table.counts()
        n_left = counts["pending"] + counts["error"]

        st.subheader("Full run")
        f = cfg.filters
        filter_text = ""
        if f:
            filter_text = (
                f" · {', '.join(str(y) for y in f.get('years', []))}"
                f" · {', '.join(f.get('countries', []))} · {f.get('section_name', '')}"
            )
        st.caption(
            f"{match_pricing(cfg.model)['label']} · {cfg.thinking_effort} effort · "
            f"{counts['total']:,} rows · config {cfg.config_id}{filter_text}"
        )
        st.caption(
            "Bookmark this page's link to come back to the run (it keeps going if you close the tab). "
            "Download regularly: a server restart loses anything not downloaded."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending", f"{counts['pending']:,}")
        c2.metric("Success", f"{counts['success']:,}")
        c3.metric("Error", f"{counts['error']:,}")
        c4.metric("Jev verified", f"{counts['jev']['verified']:,} / {counts['success']:,}")

        est = _run_estimates(run)
        concurrency = suggested_concurrency(est["avg_input"], est["avg_output"], est["avg_latency_s"], run.limits)
        jev_left = len(run.table.rowids_for_jev())

        if n_left:
            full = estimate_full_run(
                n_left, est["avg_input"], est["avg_output"], cfg.model,
                est["avg_latency_s"], concurrency, run.limits,
            )
            st.markdown(
                f"**Estimate timeframe for {n_left:,} rows:** "
                f"{_fmt_minutes(full['minutes_low'])} – {_fmt_minutes(full['minutes_high'])}"
            )
            if not run.running:
                _long_run_warning(full, st.session_state.jev_api_key_valid)

        jev_on = st.session_state.jev_api_key_valid
        has_work = n_left > 0 or (jev_on and jev_left > 0)
        b1, b2, b3 = st.columns(3)
        fresh = counts["success"] == 0 and counts["error"] == 0
        if b1.button(
            "Start" if fresh else "Resume", type="primary", use_container_width=True,
            disabled=run.running or not has_work or not st.session_state.api_key_valid,
        ):
            _confirm_full_run_dialog(n_left, jev_left, concurrency)
        if b2.button("Stop", use_container_width=True, disabled=not run.running):
            run.cancel.set()
            st.toast("Stopping: requests in flight will finish, nothing new is sent.")
        if b3.button("Close run", use_container_width=True, disabled=run.running):
            batch_processor.remove_run(token)
            st.session_state.batch_run_token = None
            st.session_state.batch_snapshot = None
            if "run" in st.query_params:
                del st.query_params["run"]
            st.rerun()

        if not st.session_state.api_key_valid:
            st.info("Validate your Claude API key in Setup to start or resume.")
        if not jev_on:
            st.caption("No Jev key set. Add one in Setup to verify results.")
        if counts["error"] and not run.running:
            st.caption(
                f"{counts['error']:,} rows failed classification and are skipped by Jev. "
                "Press Resume to retry them."
            )

        if run.running:
            st.session_state.batch_was_running = True
            _render_progress_live(run)
        else:
            _render_progress(run)

        st.divider()
        st.subheader("Downloads")
        d1, d2 = st.columns(2)
        if d1.button("Prepare download", use_container_width=True):
            _make_snapshot(run)
        snap = st.session_state.batch_snapshot
        if snap and snap["token"] == token:
            d2.download_button(
                "Download results CSV", snap["data"],
                file_name=f"full_run_{cfg.config_id}.csv", mime="text/csv",
                type="primary", use_container_width=True,
                on_click=lambda v=snap["version"]: setattr(run, "downloaded_version", v),
            )
        not_downloaded = run.table.version - max(run.downloaded_version, 0)
        if run.table.version and not_downloaded > 0:
            st.caption(
                f"{not_downloaded:,} row updates not downloaded yet. "
                "Press Prepare download for the latest snapshot."
            )
        st.caption('Editing in Excel? Save as "CSV UTF-8" before uploading again.')

        import pandas as pd

        errors = run.table.error_rows()
        if errors:
            with st.expander(f"Errors ({len(errors):,})"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Company": e["company_name"], "Year": e["year"],
                            "Status": e["status"], "Error": e["error"],
                            "Jev status": e["jev_status"], "Jev error": e["jev_error"],
                        }
                        for e in errors
                    ]),
                    use_container_width=True, hide_index=True,
                )

        if counts["success"] and not run.running:
            st.subheader("Results preview")
            status_filter = st.selectbox("Show", ["All", "success", "error", "pending"], key="batch_preview_filter")
            preview_rows = run.table.snapshot()
            if status_filter != "All":
                preview_rows = [r for r in preview_rows if r["status"] == status_filter]
            hidden = {f"{c}_quote" for c in cfg.taxonomy_codes} | {f"{c}_supports" for c in cfg.taxonomy_codes}
            df = pd.DataFrame(preview_rows, columns=[c for c in run.table.columns if c not in hidden]).astype(str)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Quote and supports columns are in the CSV download.")
