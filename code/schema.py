"""Schema, vocabulary, and normalization for the support-triage agent.

Defines the closed enums (`status`, `request_type`, `company`) per the
problem statement, derives the observed `product_area` vocabulary from
``support_tickets/sample_support_tickets.csv``, and provides validators
and coercion helpers.

Internal canonical form is lowercase. The sample CSV uses TitleCase for
``Status`` (``Replied``/``Escalated``) and lowercase for everything else,
so output formatting maps the canonical form back to the sample's mixed
convention to maximize scoring agreement.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Closed enums (canonical lowercase form). Source: problem_statement.md.
# ---------------------------------------------------------------------------

STATUS: tuple[str, ...] = ("replied", "escalated")
REQUEST_TYPE: tuple[str, ...] = ("product_issue", "feature_request", "bug", "invalid")
COMPANY: tuple[str, ...] = ("hackerrank", "claude", "visa", "none")

# Output mapping: how each canonical value is written to the CSV. The sample
# CSV uses TitleCase for Status and lowercase for everything else.
STATUS_OUTPUT_FORM = {"replied": "Replied", "escalated": "Escalated"}
REQUEST_TYPE_OUTPUT_FORM = {v: v for v in REQUEST_TYPE}  # lowercase passthrough


# ---------------------------------------------------------------------------
# product_area: open vocabulary, seeded from labeled samples.
# ---------------------------------------------------------------------------
# We treat product_area as semi-open: the agent should prefer values it has
# seen for the same company, but may emit a new snake_case value when the
# retrieved corpus clearly indicates one. Validation here is informational
# (a `is_observed` check), not enforced rejection.

PRODUCT_AREA_OBSERVED: dict[str, set[str]] = {
    "hackerrank": {"screen", "community"},
    "claude": {"privacy"},
    "visa": {"travel_support", "general_support"},
    "none": {"conversation_management"},
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _strip_lower(s: str | None) -> str:
    if s is None:
        return ""
    return s.strip().lower()


def normalize_company(raw: str | None) -> str:
    """Map any case/whitespace variant to one of COMPANY. Returns 'none' on missing."""
    s = _strip_lower(raw)
    if s in ("", "none", "n/a", "null"):
        return "none"
    if s in COMPANY:
        return s
    raise ValueError(f"unknown company: {raw!r}")


def normalize_status(raw: str | None) -> str:
    """Map any case to canonical lowercase status. Raises on unknown."""
    s = _strip_lower(raw)
    if s in STATUS:
        return s
    raise ValueError(f"unknown status: {raw!r}")


def normalize_request_type(raw: str | None) -> str:
    """Map any case to canonical lowercase request_type. Raises on unknown."""
    s = _strip_lower(raw)
    if s in REQUEST_TYPE:
        return s
    raise ValueError(f"unknown request_type: {raw!r}")


_PRODUCT_AREA_RE = re.compile(r"[^a-z0-9]+")


def normalize_product_area(raw: str | None) -> str:
    """Snake_case-lower a product_area label. Empty input returns ''.

    Does NOT raise on unknown values — product_area is open vocabulary.
    """
    if raw is None:
        return ""
    s = raw.strip().lower()
    if not s:
        return ""
    s = _PRODUCT_AREA_RE.sub("_", s).strip("_")
    return s


def is_observed_product_area(company: str, product_area: str) -> bool:
    """True if ``product_area`` was seen for ``company`` in the labeled set."""
    return product_area in PRODUCT_AREA_OBSERVED.get(company, set())


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_status(canonical: str) -> str:
    """Canonical lowercase status -> CSV-output form (TitleCase)."""
    if canonical not in STATUS_OUTPUT_FORM:
        raise ValueError(f"unknown canonical status: {canonical!r}")
    return STATUS_OUTPUT_FORM[canonical]


def format_request_type(canonical: str) -> str:
    if canonical not in REQUEST_TYPE_OUTPUT_FORM:
        raise ValueError(f"unknown canonical request_type: {canonical!r}")
    return REQUEST_TYPE_OUTPUT_FORM[canonical]


# ---------------------------------------------------------------------------
# Row validation / coercion
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ("Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type")


def validate_output_row(row: dict) -> list[str]:
    """Return a list of problems with ``row``. Empty list = valid.

    ``row`` is in the CSV-emission form (TitleCase status, etc).
    """
    problems: list[str] = []
    try:
        normalize_status(row.get("Status"))
    except ValueError as e:
        problems.append(f"Status: {e}")
    try:
        normalize_request_type(row.get("Request Type"))
    except ValueError as e:
        problems.append(f"Request Type: {e}")
    # Company is allowed to be missing/None; just check it normalizes.
    try:
        normalize_company(row.get("Company"))
    except ValueError as e:
        problems.append(f"Company: {e}")
    # Response and Product Area are open-form strings — only check presence.
    if row.get("Response") is None:
        problems.append("Response: missing")
    return problems


def coerce_to_escalated(row: dict, reason: str) -> dict:
    """Return a row coerced to a safe escalated state, preserving inputs."""
    out = dict(row)
    out["Status"] = format_status("escalated")
    out["Response"] = (
        "Thank you for reaching out. We're escalating this to a human agent "
        "who will follow up with you shortly."
    )
    out["Product Area"] = out.get("Product Area") or ""
    # Best-effort request_type preservation; default to invalid if unknown.
    try:
        out["Request Type"] = format_request_type(normalize_request_type(out.get("Request Type")))
    except ValueError:
        out["Request Type"] = format_request_type("invalid")
    out["_coerced_reason"] = reason  # internal-only, drop before CSV write
    return out


# ---------------------------------------------------------------------------
# Sample CSV loader (used by eval; lives here so the schema owns the column names)
# ---------------------------------------------------------------------------


def load_labeled_csv(path: str | Path) -> list[dict]:
    """Load ``sample_support_tickets.csv`` with normalized canonical values.

    Returns a list of dicts with keys:
    ``issue, subject, company, response, product_area, status, request_type``
    (lowercase keys; canonical lowercase values for the closed enums).
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "issue": (raw.get("Issue") or "").strip(),
                    "subject": (raw.get("Subject") or "").strip(),
                    "company": normalize_company(raw.get("Company")),
                    "response": raw.get("Response") or "",
                    "product_area": normalize_product_area(raw.get("Product Area")),
                    "status": normalize_status(raw.get("Status")),
                    "request_type": normalize_request_type(raw.get("Request Type")),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Self-test (run as `python code/schema.py`)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    # Round-trip the labeled sample CSV.
    here = Path(__file__).resolve().parent
    sample_path = here.parent / "support_tickets" / "sample_support_tickets.csv"
    rows = load_labeled_csv(sample_path)
    assert rows, "sample CSV loaded zero rows"
    for r in rows:
        assert r["status"] in STATUS, r
        assert r["request_type"] in REQUEST_TYPE, r
        assert r["company"] in COMPANY, r

    # Status formatting round-trip.
    assert format_status("replied") == "Replied"
    assert format_status("escalated") == "Escalated"
    assert normalize_status("Replied") == "replied"
    assert normalize_status("  ESCALATED  ") == "escalated"

    # Company normalization is forgiving.
    assert normalize_company("None") == "none"
    assert normalize_company("None ") == "none"  # trailing space (real in sample CSV)
    assert normalize_company(None) == "none"
    assert normalize_company("HackerRank") == "hackerrank"

    # product_area normalization.
    assert normalize_product_area("Screen") == "screen"
    assert normalize_product_area("travel support") == "travel_support"
    assert normalize_product_area("") == ""
    assert normalize_product_area(None) == ""

    # is_observed_product_area: known and unknown.
    assert is_observed_product_area("hackerrank", "screen")
    assert not is_observed_product_area("hackerrank", "fictional_area")

    # Validation flags missing/invalid fields.
    bad = {"Status": "wat", "Request Type": "nope", "Company": "Mars", "Response": None}
    problems = validate_output_row(bad)
    assert len(problems) == 4, problems

    good = {
        "Status": "Replied",
        "Request Type": "product_issue",
        "Company": "HackerRank",
        "Response": "ok",
        "Product Area": "screen",
    }
    assert validate_output_row(good) == []

    # Coerce invalid -> escalated keeps Issue/Subject untouched.
    coerced = coerce_to_escalated(
        {"Issue": "I'm angry", "Subject": "rage", "Status": "WAT", "Request Type": "????", "Response": ""},
        reason="invalid_status",
    )
    assert coerced["Status"] == "Escalated"
    assert coerced["Request Type"] == "invalid"
    assert coerced["Issue"] == "I'm angry"
    assert "human agent" in coerced["Response"].lower()

    # Print the observed vocabulary so a developer running this gets a quick view.
    print("OK: schema self-test passed")
    print(f"  rows loaded: {len(rows)}")
    print(f"  status enum: {STATUS}")
    print(f"  request_type enum: {REQUEST_TYPE}")
    print(f"  company enum: {COMPANY}")
    print(f"  product_area observed: {dict((k, sorted(v)) for k, v in PRODUCT_AREA_OBSERVED.items())}")


if __name__ == "__main__":
    _self_test()
