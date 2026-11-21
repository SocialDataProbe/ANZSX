import libsql

from config import TURSO_AUTH_TOKEN, TURSO_DATABASE_URL

SECTION_DESCRIPTIONS = {
    "Business Review & Management Commentary": "Narrative discussion of operations, financial performance, strategy, risks, governance, outlook.",
    "Directors Report": "Statutory report — names of directors, principal activities, review of operations, significant events.",
    "Remuneration Report": "Key Management Personnel compensation — remuneration policies, executive/director pay, incentives.",
    "Financial Statements": "Primary financial statements — Profit or Loss, Balance Sheet, Changes in Equity, Cash Flows.",
    "Notes to Financial Statements": "Supporting disclosures — accounting policies, breakdowns, judgments, estimates.",
    "Directors' Declaration": "Formal compliance and financial position statement by the directors.",
    "Auditor's Report": "Independent auditor's opinion on the financial statements.",
    "Other Information": "Supplementary sections — shareholder info, corporate directory, appendices.",
}


# libsql has no row_factory, so rows are wrapped to behave like sqlite3.Row:
# r[0], r["col"] and dict(r), which is all this module uses.
class _Row:
    __slots__ = ("_cols", "_vals")

    def __init__(self, cols: list[str], vals: tuple):
        self._cols = cols
        self._vals = vals

    def keys(self) -> list[str]:
        return list(self._cols)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._vals[self._cols.index(key)]
        return self._vals[key]

    def __len__(self) -> int:
        return len(self._vals)


class _Cursor:
    def __init__(self, cur):
        self._cur = cur
        self._cols = [d[0] for d in cur.description] if cur.description else []

    def fetchone(self) -> _Row | None:
        r = self._cur.fetchone()
        return None if r is None else _Row(self._cols, r)

    def fetchall(self) -> list[_Row]:
        return [_Row(self._cols, r) for r in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class _TursoConnection:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()) -> _Cursor:
        return _Cursor(self._conn.execute(sql, tuple(params)))

    def close(self):
        self._conn.close()


def _get_connection() -> _TursoConnection:
    # Only a remote Turso URL is accepted; libsql would also open a local file path.
    if not TURSO_DATABASE_URL.startswith(("libsql://", "https://")) or not TURSO_AUTH_TOKEN:
        raise RuntimeError(
            "The Turso database isn't set up. Add TURSO_DATABASE_URL (libsql://...) and "
            "TURSO_AUTH_TOKEN to .streamlit/secrets.toml, or to the app's Secrets on Streamlit Cloud."
        )
    return _TursoConnection(libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN))


def get_filter_options() -> dict:
    conn = _get_connection()
    try:
        years = [r[0] for r in conn.execute(
            "SELECT DISTINCT year FROM fts_reports ORDER BY year"
        )]
        countries = [r[0] for r in conn.execute(
            "SELECT DISTINCT country FROM fts_reports ORDER BY country"
        )]
        sections = [r[0] for r in conn.execute(
            "SELECT DISTINCT section_name FROM fts_reports ORDER BY section_name"
        )]
        # One (company, year, country) row per company-year. The Filter tab narrows
        # the company list by year and country from this, without a new query.
        company_index = [(r[0], r[1], r[2]) for r in conn.execute(
            "SELECT DISTINCT company_name, year, country FROM fts_reports ORDER BY company_name"
        )]
        total = conn.execute("SELECT COUNT(*) FROM fts_reports").fetchone()[0]
        return {
            "years": years,
            "countries": countries,
            "sections": sections,
            "company_index": company_index,
            "section_descriptions": SECTION_DESCRIPTIONS,
            "total_records": total,
        }
    finally:
        conn.close()


def _scope_where(
    years: list,
    countries: list,
    companies: list | None,
    section_name: str,
    alias: str = "",
) -> tuple[str, list]:
    """WHERE conditions (without "WHERE") for one section. No companies = all companies."""
    p = f"{alias}." if alias else ""
    parts = [
        f"{p}year IN ({','.join('?' * len(years))})",
        f"{p}country IN ({','.join('?' * len(countries))})",
    ]
    params = list(years) + list(countries)
    if companies:
        parts.append(f"{p}company_name IN ({','.join('?' * len(companies))})")
        params += list(companies)
    parts += [f"{p}section_name = ?", f"length({p}content) > 0"]
    params.append(section_name)
    return " AND ".join(parts), params


def _build_primary_where(
    years: list,
    countries: list,
    section_name: str,
    companies: list | None = None,
) -> tuple[str, list]:
    where, params = _scope_where(years, countries, companies, section_name)
    return f"SELECT {{cols}} FROM fts_reports WHERE {where}", params


# The single copy of the fallback filter, used by every function below.
def _build_fallback_where(
    years: list,
    countries: list,
    section_name: str,
    fallback_section: str,
    companies: list | None = None,
) -> tuple[str, list]:
    """Rows of the fallback section for companies that have no non-empty primary section."""
    outer, outer_params = _scope_where(years, countries, companies, fallback_section, alias="f")
    inner, inner_params = _scope_where(years, countries, companies, section_name)
    sql = (
        f"SELECT {{cols}} FROM fts_reports f WHERE {outer} "
        f"AND f.company_name NOT IN (SELECT company_name FROM fts_reports WHERE {inner})"
    )
    return sql, outer_params + inner_params


def _use_fallback(section_name: str, fallback_section: str | None) -> bool:
    return bool(fallback_section) and fallback_section != section_name


def get_filtered_count(
    years: list[str],
    countries: list[str],
    section_name: str,
    fallback_section: str | None = None,
    companies: list[str] | None = None,
) -> dict:
    conn = _get_connection()
    try:
        sql_tpl, params = _build_primary_where(years, countries, section_name, companies)
        primary_count = conn.execute(sql_tpl.format(cols="COUNT(*)"), params).fetchone()[0]

        fallback_count = 0
        if _use_fallback(section_name, fallback_section):
            fb_tpl, fb_params = _build_fallback_where(
                years, countries, section_name, fallback_section, companies
            )
            fallback_count = conn.execute(fb_tpl.format(cols="COUNT(*)"), fb_params).fetchone()[0]

        return {
            "count": primary_count + fallback_count,
            "primary_count": primary_count,
            "fallback_count": fallback_count,
        }
    finally:
        conn.close()


def get_filtered_records(
    years: list[str],
    countries: list[str],
    section_name: str,
    fallback_section: str | None = None,
    limit: int = 100,
    offset: int = 0,
    companies: list[str] | None = None,
) -> dict:
    conn = _get_connection()
    try:
        cols = "rowid, company_name, year, section_name, country, length(content) as content_length"
        sql_tpl, params = _build_primary_where(years, countries, section_name, companies)
        select_sql = sql_tpl.format(cols=cols)

        if _use_fallback(section_name, fallback_section):
            fb_tpl, fb_params = _build_fallback_where(
                years, countries, section_name, fallback_section, companies
            )
            combined = (
                f"SELECT * FROM ({select_sql} UNION ALL {fb_tpl.format(cols=cols)}) "
                f"ORDER BY company_name, year "
                f"LIMIT ? OFFSET ?"
            )
            all_params = params + fb_params + [limit, offset]
        else:
            combined = select_sql + " ORDER BY company_name, year LIMIT ? OFFSET ?"
            all_params = params + [limit, offset]

        rows = conn.execute(combined, all_params).fetchall()
        records = [dict(r) for r in rows]

        count_info = get_filtered_count(
            years, countries, section_name, fallback_section, companies,
        )
        return {
            "records": records,
            "total": count_info["count"],
            "has_more": offset + limit < count_info["count"],
        }
    finally:
        conn.close()


def get_filtered_rowids(
    years: list[str],
    countries: list[str],
    section_name: str,
    fallback_section: str | None = None,
    companies: list[str] | None = None,
) -> list[dict]:
    conn = _get_connection()
    try:
        cols = "rowid, company_name, year, section_name, country"
        sql_tpl, params = _build_primary_where(years, countries, section_name, companies)
        select_sql = sql_tpl.format(cols=cols)

        if _use_fallback(section_name, fallback_section):
            fb_tpl, fb_params = _build_fallback_where(
                years, countries, section_name, fallback_section, companies
            )
            combined = (
                f"SELECT * FROM ({select_sql} UNION ALL {fb_tpl.format(cols=cols)}) "
                f"ORDER BY company_name, year"
            )
            all_params = params + fb_params
        else:
            combined = select_sql + " ORDER BY company_name, year"
            all_params = params

        return [dict(r) for r in conn.execute(combined, all_params).fetchall()]
    finally:
        conn.close()


def get_records_by_rowids(rowids: list[int], chunk_size: int = 50) -> dict[int, dict]:
    conn = _get_connection()
    try:
        out = {}
        for i in range(0, len(rowids), chunk_size):
            chunk = rowids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                "SELECT rowid, company_name, year, section_name, country, content "
                f"FROM fts_reports WHERE rowid IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                out[r["rowid"]] = dict(r)
        return out
    finally:
        conn.close()


def build_fts_query(terms: list[str]) -> str:
    """An FTS5 query that matches any of the terms. Each term is quoted, so punctuation
    (for example the hyphen in "share-based") cannot cause a syntax error."""
    seen = set()
    quoted = []
    for term in terms:
        t = " ".join(str(term).split())
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        quoted.append('"' + t.replace('"', '""') + '"')
    return " OR ".join(quoted)


def search_ranked(
    fts_query: str,
    years: list[str],
    countries: list[str],
    section_name: str,
    limit: int,
    companies: list[str] | None = None,
) -> list[dict]:
    """Rows of the primary section that match the query, best BM25 match first.
    Returns metadata only, not content. The fallback section is not searched."""
    if not fts_query:
        return []
    conn = _get_connection()
    try:
        where, params = _scope_where(years, countries, companies, section_name)
        rows = conn.execute(
            "SELECT rowid, company_name, year, section_name, country FROM fts_reports "
            f"WHERE fts_reports MATCH ? AND {where} "
            "ORDER BY bm25(fts_reports) LIMIT ?",
            [fts_query] + params + [int(limit)],
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_record_content(rowid: int) -> dict:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT rowid, company_name, year, section_name, country, content "
            "FROM fts_reports WHERE rowid = ?",
            [rowid],
        ).fetchone()
        if not row:
            return {"error": True, "message": f"Record {rowid} not found"}
        d = dict(row)
        d["content_length"] = len(d.get("content", ""))
        d["estimated_tokens"] = d["content_length"] // 4
        return d
    finally:
        conn.close()


def get_sample_records(
    years: list[str],
    countries: list[str],
    section_name: str,
    fallback_section: str | None = None,
    n: int = 15,
    companies: list[str] | None = None,
) -> dict:
    conn = _get_connection()
    try:
        cols = "rowid, company_name, year, section_name, country, content, length(content) as content_length"
        sql_tpl, params = _build_primary_where(years, countries, section_name, companies)
        select_sql = sql_tpl.format(cols=cols)

        if _use_fallback(section_name, fallback_section):
            fb_tpl, fb_params = _build_fallback_where(
                years, countries, section_name, fallback_section, companies
            )
            combined = (
                f"SELECT * FROM ({select_sql} UNION ALL {fb_tpl.format(cols=cols)}) "
                f"ORDER BY RANDOM() LIMIT ?"
            )
            all_params = params + fb_params + [n]
        else:
            combined = select_sql + " ORDER BY RANDOM() LIMIT ?"
            all_params = params + [n]

        rows = conn.execute(combined, all_params).fetchall()
        samples = []
        for r in rows:
            d = dict(r)
            d["estimated_tokens"] = d["content_length"] // 4
            samples.append(d)
        return {"samples": samples}
    finally:
        conn.close()


def get_content_stats(
    years: list[str],
    countries: list[str],
    section_name: str,
    fallback_section: str | None = None,
    companies: list[str] | None = None,
) -> dict:
    conn = _get_connection()
    try:
        sql_tpl, params = _build_primary_where(years, countries, section_name, companies)
        stats_sql = sql_tpl.format(
            cols="COUNT(*) as cnt, AVG(length(content)) as avg_len, "
            "SUM(length(content)) as total_len, "
            "MIN(length(content)) as min_len, MAX(length(content)) as max_len"
        )
        row = conn.execute(stats_sql, params).fetchone()
        result = dict(row)

        if _use_fallback(section_name, fallback_section):
            fb_tpl, fb_params = _build_fallback_where(
                years, countries, section_name, fallback_section, companies
            )
            fb_sql = fb_tpl.format(
                cols="COUNT(*) as cnt, AVG(length(content)) as avg_len, "
                "SUM(length(content)) as total_len"
            )
            fb = dict(conn.execute(fb_sql, fb_params).fetchone())
            total_cnt = (result["cnt"] or 0) + (fb["cnt"] or 0)
            total_len = (result["total_len"] or 0) + (fb["total_len"] or 0)
            result["cnt"] = total_cnt
            result["total_len"] = total_len
            result["avg_len"] = total_len / total_cnt if total_cnt > 0 else 0

        return {
            "count": result["cnt"] or 0,
            "avg_length": result["avg_len"] or 0,
            "total_length": result["total_len"] or 0,
            "min_length": result.get("min_len") or 0,
            "max_length": result.get("max_len") or 0,
        }
    finally:
        conn.close()
