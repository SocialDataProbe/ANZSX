import asyncio
import concurrent.futures
import json
import re
import traceback

import streamlit as st

from backend import batch_processor, classifier, database
from backend.batch_table import BatchTable, RunConfig
from backend.cost import estimate_cost, estimate_full_run, match_pricing, suggested_concurrency, USD_TO_AUD
from backend.rate_limiter import start_tier_limits
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
    "selected_companies": [],
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
    # Sample Run (tab 4). Targeted mode: find hits with a keyword (BM25) search.
    "sample_mode": "Random",
    "hit_field": "",
    "hit_values": [],
    "hits_to_find": 15,
    "max_reports": 50,
    "targeted_search": None,  # hidden: search key, FTS query and rowids already read
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
    # Includes "company_index" (company, year, country) for the Companies filter.
    return database.get_filter_options()


# The load adds a spinner element, but only on the first run. The container is drawn on
# every run, so the tabs below stay at the same place. If the tabs moved, Streamlit
# would make new tabs and the user would go back to the first tab (Description).
with st.container():
    # An older session config has no company list, so load it again.
    if st.session_state.config is None or "company_index" not in st.session_state.config:
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
        "companies": st.session_state.selected_companies,
        "section_name": st.session_state.selected_section,
        "fallback_section": st.session_state.fallback_section,
    }


def _companies_text(companies) -> str:
    if not companies:
        return "All"
    if len(companies) > 5:
        return f"{len(companies):,} companies"
    return ", ".join(companies)


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


def _current_prompt() -> dict:
    # Same shape as st.session_state.saved_prompt, so the two can be compared.
    return {
        "task": st.session_state.task_prompt,
        "rules": _get_rules(),
        "taxonomy": _get_taxonomy(),
    }


# A template blank, e.g. "[the item]". Templates mark each part the user must replace.
_BLANK_RE = re.compile(r"\[[^\[\]\n]+\]")


def _find_blanks(prompt: dict) -> list[str]:
    texts = [prompt["task"], *prompt["rules"]]
    for entry in prompt["taxonomy"]:
        texts.append(entry["definition"])
        for cat in entry.get("categories", []):
            texts += [cat["name"], cat["description"]]
    found = []
    for text in texts:
        for blank in _BLANK_RE.findall(text):
            if blank not in found:
                found.append(blank)
    return found


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


TARGETED_COST_WARNING = (
    "A larger maximum reads more reports and costs more. "
    "The keyword search reads the most likely reports first."
)
TARGETED_SCOPE_WARNING = (
    "This is a top-N search. It does not read all the reports, so do not use these "
    "results to make conclusions about the full dataset."
)


def _hit_text() -> str:
    """The hit definition in words, for the confirm dialog."""
    field = st.session_state.hit_field
    values = st.session_state.hit_values
    entry = next((e for e in _get_taxonomy() if e["code"] == field), None)
    if entry and entry.get("categories"):
        return f"`{field}` is {' or '.join(values) if values else '—'}"
    return f"`{field}` has an answer (not \"Don't Know\")"


@st.dialog("Confirm Sample Run", width="large")
def _confirm_trial_dialog():
    fp = _filter_params()
    cp = _classify_params()
    targeted = st.session_state.sample_mode == "Targeted"

    st.markdown("**Are you sure you want to run the sample?**")
    st.markdown("The following configuration will be used:")

    c1, c2 = st.columns(2)
    c1.markdown(f"**Years:** {', '.join(str(y) for y in fp['years'])}")
    c2.markdown(f"**Countries:** {', '.join(fp['countries'])}")
    c1.markdown(f"**Primary Section:** {fp['section_name']}")
    if targeted:
        c2.markdown("**Fallback Section:** not used in Targeted mode")
    else:
        c2.markdown(f"**Fallback Section:** {fp.get('fallback_section') or '—'}")
    c1.markdown(f"**Companies:** {_companies_text(fp.get('companies'))}")
    c2.markdown(f"**Sample type:** {st.session_state.sample_mode}")
    if targeted:
        st.markdown(
            f"**Hit:** {_hit_text()}  \n"
            f"**Hits to find:** {st.session_state.hits_to_find} · "
            f"**Maximum reports to read:** {st.session_state.max_reports}"
        )
        st.caption(TARGETED_COST_WARNING)

    st.divider()
    col_yes, col_no = st.columns(2)
    if col_yes.button("Yes, run sample", type="primary", use_container_width=True):
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


def _set_cost_estimate(samples: list[dict], cp: dict):
    """The tab 5 estimate: average tokens of the successful samples x the filtered count."""
    successful = [s for s in samples if s["status"] == "success"]
    if not successful:
        return
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


def _classify_samples(samples: list[dict], cp: dict, spinner: str) -> dict | None:
    """Classify the given samples, run Jev, and update the cost estimate."""
    with st.spinner(spinner):
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
        _set_cost_estimate(result["samples"], cp)
    return result


def _random_sample(fp: dict, cp: dict, spinner: str) -> dict | None:
    sample_result = database.get_sample_records(
        fp["years"], fp["countries"], fp["section_name"], fp.get("fallback_section"), n=15,
        companies=fp.get("companies"),
    )
    samples = sample_result["samples"]
    if not samples:
        st.error("No records match the current filters.")
        return None
    return _classify_samples(samples, cp, spinner)


def _search_key(fp: dict, cp: dict, field: str, values: list[str]) -> str:
    """Two Targeted runs with the same key use the same ranked list."""
    return json.dumps(
        [fp["years"], fp["countries"], fp.get("companies") or [], fp["section_name"],
         cp["task"], field, sorted(values)],
        default=str,
    )


def _targeted_sample(fp: dict, cp: dict, continue_search: bool) -> dict | None:
    """Find hits with the keyword search. continue_search=True (New Sample) skips the
    reports that the last search with the same settings already read."""
    field = next((e for e in cp["taxonomy"] if e["code"] == st.session_state.hit_field), None)
    if field is None:
        st.error("Select a hit field.")
        return None
    values = list(st.session_state.hit_values) if field.get("categories") else []
    if field.get("categories") and not values:
        st.error("Select at least one hit value.")
        return None
    n_hits = int(st.session_state.hits_to_find)
    max_reports = int(st.session_state.max_reports)
    key = _search_key(fp, cp, field["code"], values)
    prev = st.session_state.targeted_search

    with st.spinner("Searching for hits... This can take a few minutes."):
        if continue_search and prev and prev["key"] == key:
            fts_query, skip = prev["fts_query"], list(prev["read_rowids"])
        else:
            terms = _run_async(classifier.make_search_terms(
                cp["task"], field, values, api_key=st.session_state.api_key,
            ))
            fts_query, skip = database.build_fts_query(terms), []
        try:
            ranked = database.search_ranked(
                fts_query, fp["years"], fp["countries"], fp["section_name"],
                limit=len(skip) + max_reports, companies=fp.get("companies"),
            )
        except Exception:
            traceback.print_exc()
            st.error("The keyword search failed. Please try again.")
            return None
        result = _run_async(classifier.run_targeted(
            ranked, field["code"], values, n_hits, max_reports,
            cp["model"], cp["task"], cp["rules"],
            taxonomy=cp["taxonomy"],
            thinking_effort=cp["thinking_effort"],
            api_key=st.session_state.api_key,
            trial_store=st.session_state.trial_store,
            get_records=database.get_records_by_rowids,
            skip_rowids=skip,
        ))

    trial = st.session_state.trial_store[result["trial_id"]]
    st.session_state.targeted_search = {
        "key": key,
        "fts_query": fts_query,
        "read_rowids": trial["read_rowids"],
    }
    result["mode"] = "Targeted"
    _run_jev_if_enabled(result, trial["raw_samples"], cp)
    # Every report read costs money, so the estimate uses hits and misses.
    _set_cost_estimate(result["all_results"], cp)
    return result


# ── Tabs ───────────────────────────────────────────────────────────

tab_bg, tab_setup, tab_filter, tab_prompt, tab_trial, tab_cost, tab_full = st.tabs([
    "Description",
    "(1) Setup",
    "(2) Filter Records",
    "(3) Extraction Prompt",
    "(4) Sample Run",
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
        "2. **Filter** — Select which years, countries, companies, and report sections to process.\n"
        "3. **Prompt** — Choose what you want from each report (extract data, classify, "
        "summarise or custom). Load an example or a template, then change it for your research.\n"
        "4. **Cost Estimate** — See the estimated token usage and cost before committing.\n"
        "5. **Sample Run** — Test your prompt on 15 random reports, or use a keyword search "
        "to find reports with a value you select (Targeted), to check quality.\n"
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

    # The company list follows the Years and Countries above (no selection = any).
    sel_years = set(st.session_state.selected_years)
    sel_countries = set(st.session_state.selected_countries)
    company_options = sorted({
        name for name, year, country in config.get("company_index", [])
        if (not sel_years or year in sel_years)
        and (not sel_countries or country in sel_countries)
    })
    if "company_ms" not in st.session_state:
        st.session_state["company_ms"] = list(st.session_state.selected_companies)
    # Remove selected companies that the Years/Countries no longer include.
    allowed = set(company_options)
    st.session_state["company_ms"] = [c for c in st.session_state["company_ms"] if c in allowed]
    st.multiselect(
        "Companies", company_options, key="company_ms",
        placeholder="All companies – type to search",
        help="Type part of a name to search. Leave empty to include all companies.",
    )
    st.caption(f"{len(company_options):,} companies match the selected years and countries.")
    st.session_state.selected_companies = list(st.session_state["company_ms"])

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
                    companies=fp.get("companies"),
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
        "Build your prompt by filling in the form below. "
        "The model automatically receives the document context (company, year, section) "
        "together with your prompt.\n\n"
        "The prompt has three parts:\n"
        "- **Task**: the overall job for each report.\n"
        "- **Output fields**: the answers you want. Each field becomes a column in your results file.\n"
        "- **Rules** *(optional)*: limits that apply to all the answers."
    )
    st.info(
        "Please click **Save Prompt** at the bottom of this tab to save your prompt. "
        "The next tabs use the saved prompt only."
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

    EXAMPLE_TASK_OUTLOOK = (
        "Summarise what the company says about its outlook and plans for the next "
        "financial year. Use only the parts of the document that discuss the outlook, "
        "future plans or strategy for the coming year."
    )

    EXAMPLE_RULES_OUTLOOK = [
        "Write no more than 3 sentences.",
        "Include only statements about the future. Do not summarise results for the current or earlier years.",
        "Do not add opinions or facts that are not in the document.",
    ]

    EXAMPLE_TAXONOMY_OUTLOOK = [
        {
            "code": "outlook_summary",
            "definition": "A summary of the company's stated outlook and plans for the next financial year, in no more than 3 sentences.",
        },
        {
            "code": "outlook_tone",
            "definition": "The overall tone of the company's statements about its outlook.",
            "categories": [
                {"name": "Positive", "description": "The company expects growth, better results or better conditions."},
                {"name": "Neutral", "description": "The company expects results or conditions to stay about the same, or gives no clear direction."},
                {"name": "Negative", "description": "The company expects lower results, worse conditions or major problems."},
                {"name": "Mixed", "description": "The company expects both clearly good and clearly bad results or conditions, and neither is the main message."},
            ],
        },
    ]

    # Templates: each part in [square brackets] is a blank for the user to replace.
    # Save warns while blanks remain (see _find_blanks).
    TEMPLATE_TASK_EXTRACT = (
        "Find [the item you want, e.g. the number of employees] in the document. "
        "Give the value for the target year only."
    )

    TEMPLATE_RULES_EXTRACT = [
        "Use only [where to look, e.g. the table titled ...].",
        "If the value appears more than once, use [which one to use].",
        "Copy the value exactly as it appears in the document. Do not calculate or convert it.",
    ]

    TEMPLATE_TAXONOMY_EXTRACT = [
        {
            "code": "value",
            "definition": "[The item] for the target year, exactly as it appears in the document.",
        },
        {
            "code": "unit",
            "definition": "The unit of [the item], as shown in the document.",
            "categories": [
                {"name": "[Unit 1]", "description": "[What this unit means]"},
                {"name": "[Unit 2]", "description": "[What this unit means]"},
            ],
        },
        {
            "code": "reported",
            "definition": "Whether the document reports [the item].",
            "categories": [
                {"name": "Yes", "description": "The document reports [the item]."},
                {"name": "No", "description": "The document does not report [the item]."},
            ],
        },
    ]

    TEMPLATE_TASK_CLASSIFY = (
        "Classify the company by [what you want to classify, e.g. its main customer type], "
        "based on [which part of the document]. Ignore all other content in the document."
    )

    TEMPLATE_RULES_CLASSIFY = [
        "Classify by [the main basis for the decision], not by [what the model must ignore].",
        "If more than one category fits, choose [how to choose, e.g. the category that best describes the main activity].",
    ]

    TEMPLATE_TAXONOMY_CLASSIFY = [
        {
            "code": "category",
            "definition": "The [thing you classify] of the company, based on [which part of the document].",
            "categories": [
                {"name": "[Category A]", "description": "[When to use this category, and where it stops]"},
                {"name": "[Category B]", "description": "[When to use this category, and where it stops]"},
                {"name": "[Category C]", "description": "[When to use this category, and where it stops]"},
            ],
        },
    ]

    TEMPLATE_TASK_SUMMARY = (
        "Summarise what the company says about [the topic, e.g. its main risks]. "
        "Use only the company's own statements in the document."
    )

    TEMPLATE_RULES_SUMMARY = [
        "Write no more than [number] sentences.",
        "Use plain language. Do not copy long passages from the document.",
        "Do not add opinions or facts that are not in the document.",
    ]

    TEMPLATE_TAXONOMY_SUMMARY = [
        {
            "code": "summary",
            "definition": "A summary of what the company says about [the topic], in no more than [number] sentences.",
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

    # The cards only fill in the form below (as _set_example always did).
    # They do not change what the app sends to the model.
    PROMPT_TYPES = [
        {
            "key": "extract",
            "title": "Extract data",
            "text": "Find a value, e.g. revenue.",
            "example": ("Revenue", "Financial Statements",
                        (EXAMPLE_TASK_REVENUE, EXAMPLE_RULES_REVENUE, EXAMPLE_TAXONOMY_REVENUE)),
            "template": (TEMPLATE_TASK_EXTRACT, TEMPLATE_RULES_EXTRACT, TEMPLATE_TAXONOMY_EXTRACT),
        },
        {
            "key": "classify",
            "title": "Classify",
            "text": "Put each company in one group from a list, e.g. industry.",
            "example": ("Industry", "Business Review",
                        (EXAMPLE_TASK_INDUSTRY, EXAMPLE_RULES_INDUSTRY, EXAMPLE_TAXONOMY_INDUSTRY)),
            "template": (TEMPLATE_TASK_CLASSIFY, TEMPLATE_RULES_CLASSIFY, TEMPLATE_TAXONOMY_CLASSIFY),
        },
        {
            "key": "summary",
            "title": "Summarise",
            "text": "Write a short text, e.g. the company's outlook.",
            "example": ("Outlook", "Business Review",
                        (EXAMPLE_TASK_OUTLOOK, EXAMPLE_RULES_OUTLOOK, EXAMPLE_TAXONOMY_OUTLOOK)),
            "template": (TEMPLATE_TASK_SUMMARY, TEMPLATE_RULES_SUMMARY, TEMPLATE_TAXONOMY_SUMMARY),
        },
    ]

    st.markdown("**What do you want to get from each report?**")
    type_cols = st.columns(4)
    for col, ptype in zip(type_cols, PROMPT_TYPES):
        with col, st.container(border=True):
            st.markdown(f"**{ptype['title']}**")
            st.caption(ptype["text"])
            ex_name, ex_section, ex_args = ptype["example"]
            if st.button(f"Example: {ex_name}", key=f"example_{ptype['key']}", use_container_width=True):
                _set_example(*ex_args)
                st.rerun()
            st.caption(f"Example section: {ex_section}")
            if st.button("Use template", key=f"template_{ptype['key']}", use_container_width=True):
                _set_example(*ptype["template"])
                st.rerun()
    with type_cols[3], st.container(border=True):
        st.markdown("**Custom (blank form)**")
        st.caption("Start with an empty form and build your own prompt.")
        if st.button("Start blank", key="template_custom", use_container_width=True):
            _set_example("", [])
            st.rerun()
    st.caption(
        "All four use the same form below. The choice only fills in the form for you. "
        "You can change anything. In a template, replace each part in [square brackets] "
        "with your own text."
    )

    st.session_state.task_prompt = st.text_area(
        "Task ✱",
        value=st.session_state.task_prompt,
        height=150,
        help=(
            "Describe the overall job for each report, e.g. what to find and where to look "
            "in the document. You do not need to describe the output format: the app makes "
            "it from your output fields. The model gets this task for every record."
        ),
        placeholder="e.g. Classify this company's primary industry based on the report content...",
    )

    if not st.session_state.task_prompt.strip():
        st.warning("A task is required before you can run a sample.")

    st.divider()

    # ── Output fields section (mandatory). Called "taxonomy" in the code. ──
    st.markdown("**Output fields** ✱ — the answers you want from each report")
    st.caption(
        "Each field becomes a column in your results file. "
        "**Column name** is the name of the column (e.g. `revenue`). "
        "**What to find** tells the model what to give for this field. "
        "You can add **Allowed answers** to make the model choose one answer from a fixed list."
    )

    for i in range(st.session_state.get("taxonomy_count", 0)):
        col_code, col_del = st.columns([6, 1])
        with col_code:
            st.text_input(
                f"Column name {i + 1}",
                key=f"tax_{i}_code",
                label_visibility="collapsed",
                placeholder="Column name (e.g. total_revenue)",
            )
        with col_del:
            if st.button(":material/close:", key=f"del_tax_{i}", help="Remove this field"):
                _delete_taxonomy_entry(i)
                st.rerun()

        code_val = st.session_state.get(f"tax_{i}_code", "")
        if code_val and not _is_valid_code(code_val):
            st.error("The column name must start with a letter and contain only letters, numbers and underscores.")

        st.text_input(
            f"What to find {i + 1}",
            key=f"tax_{i}_definition",
            label_visibility="collapsed",
            placeholder="What to find: what must the model give for this field?",
        )

        if not st.session_state.get(f"tax_{i}_show_cats"):
            if st.button("+ Add Allowed Answers", key=f"add_cats_{i}", help="Make the model choose one answer from a fixed list"):
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
                    st.markdown("**Allowed answers**")
                with col_cat_close:
                    if st.button(":material/close:", key=f"close_cats_{i}", help="Remove all allowed answers"):
                        for j in range(st.session_state.get(f"tax_{i}_cat_count", 0)):
                            for suffix in ("_name", "_desc"):
                                key = f"tax_{i}_cat_{j}{suffix}"
                                if key in st.session_state:
                                    del st.session_state[key]
                        st.session_state[f"tax_{i}_cat_count"] = 0
                        st.session_state[f"tax_{i}_show_cats"] = False
                        st.rerun()

                st.info(
                    "The model must give **exactly one** of the allowed answers. It cannot choose "
                    "more than one or write its own text. The app adds a \"Don't Know\" answer "
                    "automatically. If you do not add allowed answers, the model writes its own "
                    "text for this field.",
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
                            placeholder="Answer",
                        )
                    with col_desc:
                        st.text_input(
                            f"Cat desc {i}_{j}",
                            key=f"tax_{i}_cat_{j}_desc",
                            label_visibility="collapsed",
                            placeholder="Description: when must the model use this answer?",
                        )
                    with col_cdel:
                        if st.button(":material/close:", key=f"del_cat_{i}_{j}", help="Remove this answer"):
                            _delete_category(i, j)
                            st.rerun()

                if cat_count < 30:
                    if st.button("+ Add Answer", key=f"add_cat_{i}"):
                        st.session_state[f"tax_{i}_cat_{cat_count}_name"] = ""
                        st.session_state[f"tax_{i}_cat_{cat_count}_desc"] = ""
                        st.session_state[f"tax_{i}_cat_count"] = cat_count + 1
                        st.rerun()
                else:
                    st.caption("Maximum of 30 answers reached.")

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
        st.warning("At least one output field is required.")

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

    current = _current_prompt()
    blanks = _find_blanks(current)
    if blanks:
        shown = ", ".join(f"`{b}`" for b in blanks[:5]) + (", ..." if len(blanks) > 5 else "")
        st.warning(
            f"Your prompt still has blanks to fill in: {shown}. "
            "Replace each part in [square brackets] with your own text."
        )

    can_save = bool(current["task"].strip()) and len(current["taxonomy"]) > 0
    if st.button("Save Prompt", type="primary", disabled=not can_save):
        st.session_state.saved_prompt = current
        st.success("Prompt saved.")

    has_input = bool(current["task"].strip() or current["rules"] or current["taxonomy"])
    if has_input and st.session_state.saved_prompt != current:
        st.warning("You have changes that are not saved.")

    if st.session_state.saved_prompt:
        saved = st.session_state.saved_prompt
        rule_count = len(saved["rules"])
        rule_info = f" with {rule_count} rule{'s' if rule_count != 1 else ''}" if rule_count else ""
        tax_count = len(saved.get("taxonomy", []))
        tax_info = f", {tax_count} field{'s' if tax_count != 1 else ''}" if tax_count else ""
        st.caption(f"Saved prompt: task ({len(saved['task'])} chars){rule_info}{tax_info}")


# ── Tab 4: Sample Run ──────────────────────────────────────────────

with tab_trial:
    if not is_ready:
        st.info("Complete Setup, Filter, and Prompt tabs first.")
    else:
        saved = st.session_state.saved_prompt
        prompt_saved = saved == _current_prompt()
        if not prompt_saved:
            if saved:
                st.warning(
                    "Your prompt has changed since it was saved. Click **Save Prompt** at the "
                    "bottom of **(3) Extraction Prompt** to run a sample with the new prompt."
                )
            else:
                st.warning(
                    "Click **Save Prompt** at the bottom of **(3) Extraction Prompt** "
                    "before you run a sample."
                )

        def _keep(widget_key: str, store_key: str):
            # Streamlit removes a widget's own key when the widget is not drawn,
            # so each widget gets its start value from a key that stays.
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state[store_key]

        _keep("sample_mode_w", "sample_mode")
        st.radio(
            "Sample type", ["Random", "Targeted"], key="sample_mode_w", horizontal=True,
            captions=[
                "15 random reports from the filtered set.",
                "Reads the most likely reports first, until it finds the hits you ask for.",
            ],
        )
        st.session_state.sample_mode = st.session_state["sample_mode_w"]
        targeted = st.session_state.sample_mode == "Targeted"

        hit_ready = True
        if targeted:
            taxonomy_now = _get_taxonomy()
            codes = [e["code"] for e in taxonomy_now]
            _keep("hit_field_w", "hit_field")
            if st.session_state["hit_field_w"] not in codes:
                st.session_state["hit_field_w"] = codes[0]
            col_field, col_values = st.columns(2)
            col_field.selectbox(
                "Hit field", codes, key="hit_field_w",
                help="A report is a hit if Claude gives this field one of the hit values.",
            )
            st.session_state.hit_field = st.session_state["hit_field_w"]
            hit_entry = taxonomy_now[codes.index(st.session_state.hit_field)]
            hit_cats = [c["name"] for c in hit_entry.get("categories", [])]
            with col_values:
                if hit_cats:
                    _keep("hit_values_w", "hit_values")
                    # Remove values that this field no longer has, before the widget is drawn.
                    st.session_state["hit_values_w"] = [
                        v for v in st.session_state["hit_values_w"] if v in hit_cats
                    ]
                    st.multiselect(
                        "Hit values", hit_cats, key="hit_values_w",
                        placeholder="Select one or more values",
                    )
                    st.session_state.hit_values = list(st.session_state["hit_values_w"])
                    hit_ready = bool(st.session_state.hit_values)
                else:
                    st.caption(
                        "This field has no categories. Any answer except \"Don't Know\" "
                        "counts as a hit."
                    )
            col_hits, col_max = st.columns(2)
            _keep("hits_to_find_w", "hits_to_find")
            col_hits.number_input(
                "Hits to find", min_value=1, max_value=50, step=1, key="hits_to_find_w",
            )
            st.session_state.hits_to_find = int(st.session_state["hits_to_find_w"])
            _keep("max_reports_w", "max_reports")
            if st.session_state["max_reports_w"] < st.session_state.hits_to_find:
                st.session_state["max_reports_w"] = st.session_state.hits_to_find
            col_max.number_input(
                "Maximum reports to read", min_value=st.session_state.hits_to_find,
                max_value=200, step=1, key="max_reports_w",
                help="The search stops at this number, also if it has not found all the hits.",
            )
            st.session_state.max_reports = int(st.session_state["max_reports_w"])
            st.warning(TARGETED_COST_WARNING, icon=":material/payments:")
            st.warning(TARGETED_SCOPE_WARNING, icon=":material/warning:")
            if st.session_state.fallback_section:
                st.caption("Targeted mode searches the primary section only.")
            if not hit_ready:
                st.caption("Select at least one hit value to run the sample.")

        run_trial_clicked = st.button("Run Sample", disabled=not (prompt_saved and hit_ready))
        if not targeted:
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
            if targeted:
                result = _targeted_sample(fp, cp, continue_search=False)
            else:
                result = _random_sample(
                    fp, cp, "Running the sample on 15 random reports... This may take 30-60 seconds.",
                )
            if result:
                st.session_state.trial_result = result

        if st.session_state.trial_result:
            tr = st.session_state.trial_result
            tr_targeted = tr.get("mode") == "Targeted"
            if tr_targeted:
                if tr.get("retried"):
                    st.markdown(f"**Classified the same {len(tr['samples'])} reports again.**")
                else:
                    st.markdown(
                        f"**Read {tr['reports_read']} reports, found {tr['hits_found']} hits. "
                        f"Stopped: {tr['stop_reason']}.**"
                    )
                if not targeted:
                    st.warning(TARGETED_SCOPE_WARNING, icon=":material/warning:")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hits" if tr_targeted else "Succeeded", tr["success_count"])
            c2.metric("Errors", tr["error_count"])
            c3.metric("Sample Cost", f"A${tr['total_cost_usd'] * USD_TO_AUD:.4f}")

            col_retry, col_new = st.columns(2)
            with col_retry:
                if st.button("Retry Same", use_container_width=True, disabled=not tr["samples"]):
                    trial_data = st.session_state.trial_store.get(tr["trial_id"])
                    if trial_data:
                        result = _classify_samples(
                            trial_data["raw_samples"], _classify_params(),
                            "Running the sample again with the same reports...",
                        )
                        if result:
                            if tr_targeted:
                                result.update(mode="Targeted", retried=True)
                            st.session_state.trial_result = result
                            st.rerun()
            with col_new:
                if st.button("New Sample", use_container_width=True, disabled=targeted and not hit_ready):
                    fp = _filter_params()
                    cp = _classify_params()
                    if targeted:
                        result = _targeted_sample(fp, cp, continue_search=True)
                    else:
                        result = _random_sample(fp, cp, "Running the sample on 15 new reports...")
                    if result:
                        st.session_state.trial_result = result
                        st.rerun()

            if not tr["samples"]:
                st.info(f"No hits in {tr.get('reports_read', 0)} reports.")
            else:
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
        st.info("Run a sample first (tab 4). The cost estimate uses the actual API usage of the sample.")
    else:
        ce = st.session_state.cost_estimate
        st.caption(
            f"Based on {ce['trial_sample_count']} successful sample reports, "
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
def _cached_content_stats(years, countries, section_name, fallback_section, companies=()):
    return database.get_content_stats(
        list(years), list(countries), section_name, fallback_section or None,
        companies=list(companies),
    )


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
    # A Targeted sample shows hits only, but every report it read gives usage figures.
    return [s for s in tr.get("all_results") or tr["samples"] if s["status"] == "success"]


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
            "source": f"your sample run ({n} reports)",
        }
    f = run.config.filters
    content = _cached_content_stats(
        tuple(f.get("years", [])), tuple(f.get("countries", [])),
        f.get("section_name", ""), f.get("fallback_section", ""),
        tuple(f.get("companies", [])),
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
        "source": "default assumptions (no sample run with these settings)",
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
            ("Extraction prompt and output fields set (Extraction Prompt)", bool(
                st.session_state.task_prompt.strip() and _get_taxonomy()
            )),
            ("Sample run done (Sample Run)", st.session_state.trial_result is not None),
        ]
        for label, ok in checks:
            st.markdown(f"{'✅' if ok else '❌'} {label}")
        if st.button("Prepare run", type="primary", disabled=not all(ok for _, ok in checks)):
            fp = _filter_params()
            with st.spinner("Building the run table from your filters (can take ~30 s)..."):
                rows = database.get_filtered_rowids(
                    fp["years"], fp["countries"], fp["section_name"], fp.get("fallback_section") or None,
                    companies=fp.get("companies"),
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
            if f.get("companies"):
                filter_text += f" · {_companies_text(f['companies'])}"
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
