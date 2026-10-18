import csv
import hashlib
import io
import json
import threading
from dataclasses import dataclass, field

from backend.verification import compute_dn_confidence

STATUSES = ("pending", "success", "error")
JEV_STATUSES = ("pending", "verified", "skipped", "error")

BASE_COLUMNS = ["rowid", "company_name", "year", "country", "section_name", "status"]
META_COLUMNS = [
    "model_used", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
    "error", "config_id",
]
JEV_TAIL_COLUMNS = ["confidence_rule", "confidence_quote", "confidence_reason", "confidence_DN"]


@dataclass(frozen=True)
class RunConfig:
    model: str
    thinking_effort: str
    task: str
    rules: tuple = ()
    taxonomy: tuple = ()
    filters: dict = field(default_factory=dict, compare=False)

    @property
    def config_id(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "effort": self.thinking_effort,
                "task": self.task,
                "rules": list(self.rules),
                "taxonomy": list(self.taxonomy),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def taxonomy_codes(self) -> list[str]:
        return [e["code"] for e in self.taxonomy]

    @classmethod
    def from_params(cls, cp: dict) -> "RunConfig":
        return cls(
            model=cp["model"],
            thinking_effort=cp["thinking_effort"],
            task=cp["task"],
            rules=tuple(cp.get("rules") or []),
            taxonomy=tuple(cp.get("taxonomy") or []),
            filters={
                "years": list(cp.get("years") or []),
                "countries": list(cp.get("countries") or []),
                "section_name": cp.get("section_name", ""),
                "fallback_section": cp.get("fallback_section") or "",
            },
        )

    def columns(self) -> list[str]:
        cols = list(BASE_COLUMNS)
        for code in self.taxonomy_codes:
            cols += [code, f"{code}_quote", f"{code}_supports"]
        cols += META_COLUMNS
        cols += ["jev_status", "jev_error"]
        cols += [f"{code}_confidence" for code in self.taxonomy_codes]
        cols += JEV_TAIL_COLUMNS
        return cols


class BatchTable:
    """All rows of a run, keyed by rowid. Safe to update from the worker thread."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.columns = config.columns()
        self.rows: dict[int, dict] = {}
        self.order: list[int] = []
        self.lock = threading.Lock()
        self.version = 0

    @classmethod
    def new(cls, records: list[dict], config: RunConfig) -> "BatchTable":
        table = cls(config)
        for r in records:
            row = {c: "" for c in table.columns}
            row.update({
                "rowid": int(r["rowid"]),
                "company_name": r["company_name"],
                "year": str(r["year"]),
                "country": r.get("country", ""),
                "section_name": r["section_name"],
                "status": "pending",
                "config_id": config.config_id,
                "jev_status": "pending",
            })
            table.rows[row["rowid"]] = row
            table.order.append(row["rowid"])
        return table

    # ── Queries ──

    def rowids_to_process(self) -> list[int]:
        with self.lock:
            return [rid for rid in self.order if self.rows[rid]["status"] != "success"]

    def rowids_for_jev(self) -> list[int]:
        with self.lock:
            return [
                rid for rid in self.order
                if self.rows[rid]["status"] == "success"
                and self.rows[rid]["jev_status"] in ("pending", "error")
            ]

    def get(self, rowid: int) -> dict:
        with self.lock:
            return dict(self.rows[rowid])

    def counts(self) -> dict:
        with self.lock:
            out = {s: 0 for s in STATUSES}
            jev = {s: 0 for s in JEV_STATUSES}
            cost = 0.0
            for row in self.rows.values():
                out[row["status"]] = out.get(row["status"], 0) + 1
                if row["status"] == "success":
                    jev[row["jev_status"]] = jev.get(row["jev_status"], 0) + 1
                try:
                    cost += float(row["cost_usd"] or 0)
                except ValueError:
                    pass
            return {"total": len(self.rows), **out, "jev": jev, "cost_usd": cost}

    def usage_stats(self) -> dict | None:
        """Average tokens and latency over successful rows."""
        with self.lock:
            ins, outs, lats = [], [], []
            for row in self.rows.values():
                if row["status"] != "success":
                    continue
                try:
                    ins.append(float(row["input_tokens"]))
                    outs.append(float(row["output_tokens"]))
                    lats.append(float(row["latency_ms"]))
                except (TypeError, ValueError):
                    continue
        if not ins:
            return None
        return {
            "n": len(ins),
            "avg_input": sum(ins) / len(ins),
            "avg_output": sum(outs) / len(outs),
            "avg_latency_s": sum(lats) / len(lats) / 1000,
        }

    def error_rows(self) -> list[dict]:
        with self.lock:
            return [
                dict(self.rows[rid]) for rid in self.order
                if self.rows[rid]["status"] == "error" or self.rows[rid]["jev_status"] == "error"
            ]

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [dict(self.rows[rid]) for rid in self.order]

    # ── Updates ──

    def set_result(self, rowid: int, result: dict, meta: dict):
        with self.lock:
            row = self.rows[rowid]
            for code in self.config.taxonomy_codes:
                row[code] = result.get(code, "")
                row[f"{code}_quote"] = result.get(f"{code}_quote", "")
                row[f"{code}_supports"] = result.get(f"{code}_supports", "")
            row.update(meta)
            row["status"] = "success"
            row["error"] = ""
            row["config_id"] = self.config.config_id
            self._reset_jev(row)
            self.version += 1

    def set_error(self, rowid: int, message: str, meta: dict | None = None):
        with self.lock:
            row = self.rows[rowid]
            if meta:
                row.update(meta)
            row["status"] = "error"
            row["error"] = message
            row["config_id"] = self.config.config_id
            self.version += 1

    def set_jev(self, rowid: int, verification: dict):
        with self.lock:
            row = self.rows[rowid]
            self._reset_jev(row)
            if verification.get("skipped"):
                row["jev_status"] = "skipped"
                row["jev_error"] = verification.get("skip_reason", "")
            else:
                fc = verification.get("field_confidences", {})
                for code in self.config.taxonomy_codes:
                    if code in fc:
                        row[f"{code}_confidence"] = fc[code]["score"]
                for key in ("confidence_rule", "confidence_quote", "confidence_reason"):
                    value = verification.get(key)
                    row[key] = "" if value is None else value
                dn = compute_dn_confidence(fc)
                row["confidence_DN"] = "N/A" if dn is None else dn
                row["jev_status"] = "verified"
            self.version += 1

    def set_jev_error(self, rowid: int, message: str):
        with self.lock:
            row = self.rows[rowid]
            row["jev_status"] = "error"
            row["jev_error"] = message
            self.version += 1

    def _reset_jev(self, row: dict):
        row["jev_status"] = "pending"
        row["jev_error"] = ""
        for code in self.config.taxonomy_codes:
            row[f"{code}_confidence"] = ""
        for key in JEV_TAIL_COLUMNS:
            row[key] = ""

    # ── CSV ──

    def to_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.columns, extrasaction="ignore")
        writer.writeheader()
        for row in self.snapshot():
            writer.writerow(row)
        return buf.getvalue().encode("utf-8-sig")

    @classmethod
    def from_csv(cls, raw: bytes, config: RunConfig) -> "BatchTable":
        """Raises ValueError with a readable message if the CSV is invalid."""
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        header = reader.fieldnames or []
        table = cls(config)
        missing = [c for c in table.columns if c not in header]
        if missing:
            shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
            raise ValueError(f"CSV is missing columns for this run's settings: {shown}")

        problems = []
        for line_no, rec in enumerate(reader, start=2):
            try:
                rowid = int(str(rec["rowid"]).strip())
            except ValueError:
                problems.append(f"line {line_no}: rowid '{rec['rowid']}' is not a number")
                continue
            if rowid in table.rows:
                problems.append(f"line {line_no}: rowid {rowid} appears more than once")
                continue
            status = (rec["status"] or "").strip().lower()
            if status not in STATUSES:
                problems.append(
                    f"line {line_no}: status '{rec['status']}' must be pending, success or error"
                )
                continue
            jev_status = (rec["jev_status"] or "").strip().lower() or "pending"
            if jev_status not in JEV_STATUSES:
                problems.append(
                    f"line {line_no}: jev_status '{rec['jev_status']}' must be "
                    "pending, verified, skipped or error"
                )
                continue
            config_id = (rec["config_id"] or "").strip()
            if config_id and config_id != config.config_id:
                problems.append(
                    f"line {line_no}: config_id {config_id} doesn't match the current model and "
                    f"Extraction Prompt settings ({config.config_id})"
                )
                continue
            row = {c: (rec.get(c) or "") for c in table.columns}
            row.update({
                "rowid": rowid,
                "status": status,
                "jev_status": jev_status,
                "config_id": config.config_id,
            })
            table.rows[rowid] = row
            table.order.append(rowid)
            if len(problems) >= 10:
                break

        if problems:
            raise ValueError("CSV problems:\n- " + "\n- ".join(problems[:10]))
        if not table.rows:
            raise ValueError("CSV has no rows.")
        return table
