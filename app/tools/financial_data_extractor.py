import json
import re
from typing import Any

import httpx
import pymupdf as fitz
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from ..config import get_settings
from ..schemas import FinancialQuarter, MetricEvidence

settings = get_settings()
MANIFEST_PATH = settings.data_dir / "source_manifest.json"


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


def _download(url: str, suffix: str, cache_name: str) -> tuple[str, str]:
    """Download/cache a source and return raw HTML for HTML sources, text for PDFs."""
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{cache_name}{suffix}"

    if cache_path.exists() and cache_path.stat().st_size > 100:
        if suffix == ".pdf":
            with fitz.open(cache_path) as doc:
                return "pdf", "\n".join(page.get_text() for page in doc)
        return "html", cache_path.read_text(encoding="utf-8", errors="ignore")

    headers = {"User-Agent": "Mozilla/5.0 TCS-Forecasting-Agent/1.0"}
    with httpx.Client(
        timeout=settings.request_timeout_seconds,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type or suffix == ".pdf":
        cache_path.write_bytes(response.content)
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        return "pdf", text

    cache_path.write_text(response.text, encoding="utf-8")
    return "html", response.text


def _first_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _numbers_in_cell(cell: str) -> list[float]:
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", cell.replace("\u00a0", " "))
    return [float(v.replace(",", "")) for v in values]


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _statement_rows_from_html(html: str) -> list[list[str]]:
    """Find the quarterly IFRS income-statement table and return normalized rows.

    The official TCS pages expose the financial statement as an HTML table.
    Parsing the DOM is more reliable than applying pipe-delimited regexes to
    text because BeautifulSoup removes table separators during text cleanup.
    """
    soup = BeautifulSoup(html, "html.parser")
    best_rows: list[list[str]] = []
    best_score = -1

    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                for cell in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append(cells)

        if not rows:
            continue

        table_text = _normalise(" ".join(" ".join(r) for r in rows))
        labels = {_normalise(r[0]) for r in rows if r}
        score = 0
        if "revenue" in labels or any(x.startswith("revenue") for x in labels):
            score += 2
        if "net income" in labels:
            score += 3
        if "operating income" in labels:
            score += 2
        if "cost of revenue" in labels:
            score += 1
        if "three-month period" in table_text or "three-month periods" in table_text:
            score += 6

        if score > best_score:
            best_score = score
            best_rows = rows

    return best_rows


def _row_last_value(rows: list[list[str]], label: str) -> float | None:
    target = _normalise(label)
    for row in rows:
        if not row:
            continue
        first = _normalise(row[0])
        if first == target:
            values: list[float] = []
            for cell in row[1:]:
                values.extend(_numbers_in_cell(cell))
            if values:
                return values[-1]
    return None


def _table_metrics(html: str) -> tuple[float | None, float | None]:
    rows = _statement_rows_from_html(html)
    return _row_last_value(rows, "Revenue"), _row_last_value(rows, "Net income")


def _evidence(text: str, needles: list[str], window: int = 240) -> str:
    lower = text.lower()
    for needle in needles:
        idx = lower.find(needle.lower())
        if idx >= 0:
            left = max(0, idx - window)
            right = min(len(text), idx + len(needle) + window)
            return re.sub(r"\s+", " ", text[left:right]).strip()
    return ""


def extract_quarter(quarter: str) -> FinancialQuarter:
    manifest = _load_manifest()["quarters"]
    if quarter not in manifest:
        raise ValueError(f"Unsupported quarter: {quarter}")

    meta = manifest[quarter]
    kind, source = _download(
        meta["financial_report_url"], ".html", f"financial_{quarter}"
    )
    report_html = source if kind == "html" else ""
    report_text = _clean_html(source) if kind == "html" else source

    # Prefer the explicit three-month IFRS table for the current quarter.
    revenue_usd, net_profit_usd_mn = _table_metrics(report_html) if report_html else (None, None)

    # Fall back to the highlights section only if the table could not be parsed.
    summary = report_text
    if revenue_usd is None:
        revenue_usd = _first_float(
            [
                r"Revenue(?: at)?\s+US\$\s*([\d,]+)\s*million",
                r"Revenue(?: at)?\s*\$\s*([\d,]+)\s*(?:Mn|million)",
            ],
            summary,
        )

    revenue_inr_cr = _first_float(
        [
            r"Revenue\s+at\s+₹\s*([\d,]+)\s*crore",
            r"INR Revenue\s+of\s+₹\s*([\d,]+)\s*Mn",
        ],
        summary,
    )

    if net_profit_usd_mn is None:
        net_profit_usd_mn = _first_float(
            [
                r"Net (?:Income|income)\s+at\s+US\$\s*([\d,]+)\s*million",
                r"Net income\s+\$\s*([\d,]+)\s*(?:Mn|million)",
            ],
            summary,
        )

    operating_margin = _first_float(
        [r"Operating Margin(?: at|:)\s*([\d.]+)%"], summary
    )
    net_margin = _first_float(
        [r"Net Margin(?: at|:)\s*([\d.]+)%"], summary
    )
    tcv = _first_float(
        [
            r"TCV[^$]{0,100}US\$\s*([\d.]+)\s*billion",
            r"TCV[^$]{0,100}\$\s*([\d.]+)\s*billion",
        ],
        summary,
    )
    ai_rev = _first_float(
        [
            r"Annualized AI (?:Services )?Revenue[^$]{0,30}(?:US\$|\$)\s*([\d.]+)\s*billion",
            r"AI (?:Services )?Revenue[^$]{0,60}(?:US\$|\$)\s*([\d.]+)\s*billion",
        ],
        summary,
    )
    workforce = _first_float(
        [r"Workforce strength:\s*([\d,]+)", r"Employee Headcount:\s*([\d,]+)"],
        summary,
    )
    attrition = _first_float(
        [r"Attrition(?:\s*\([^)]*\))?[^\d]{0,30}([\d.]+)%"],
        summary,
    )

    evidence = []
    metric_rows = [
        ("Revenue", ["Revenue"], revenue_usd, "USD mn"),
        ("Revenue (INR)", ["INR Revenue"], revenue_inr_cr, "INR cr"),
        ("Net Profit", ["Net Income", "Net income"], net_profit_usd_mn, "USD mn"),
        ("Operating Margin", ["Operating Margin"], operating_margin, "%"),
        ("Net Margin", ["Net Margin"], net_margin, "%"),
        ("TCV", ["TCV"], tcv, "USD bn"),
        ("AI Annualized Revenue", ["AI Revenue", "AI Services Revenue"], ai_rev, "USD bn"),
        ("Workforce", ["Workforce strength", "Employee Headcount"], workforce, "people"),
        ("Attrition", ["Attrition"], attrition, "%"),
    ]
    for metric, needles, value, unit in metric_rows:
        if value is not None:
            evidence.append(
                MetricEvidence(
                    metric=metric,
                    value=value,
                    unit=unit,
                    period=quarter,
                    source=meta["financial_report_url"],
                    evidence=_evidence(report_text, needles),
                )
            )

    return FinancialQuarter(
        quarter=quarter,
        revenue_usd_mn=revenue_usd,
        revenue_inr_cr=revenue_inr_cr,
        net_profit_usd_mn=net_profit_usd_mn,
        operating_margin_pct=operating_margin,
        net_margin_pct=net_margin,
        tcv_usd_bn=tcv,
        ai_annualized_revenue_usd_bn=ai_rev,
        workforce=int(workforce) if workforce else None,
        attrition_pct=attrition,
        evidence=evidence,
    )


@tool("FinancialDataExtractorTool")
def financial_data_extractor_tool(quarters: list[str]) -> str:
    """Extract core financial metrics from TCS quarterly financial reports with source evidence."""
    result = [extract_quarter(q).model_dump() for q in quarters]
    return json.dumps(
        {"tool": "FinancialDataExtractorTool", "quarters": result}, indent=2
    )
