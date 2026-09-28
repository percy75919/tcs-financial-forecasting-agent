import json
import re
from typing import Any

import pymupdf as fitz
import httpx
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
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"{cache_name}{suffix}"

    if cache_path.exists() and cache_path.stat().st_size > 100:
        if suffix == ".pdf":
            with fitz.open(cache_path) as doc:
                return "pdf", "\n".join(page.get_text() for page in doc)
        return "html", _clean_html(cache_path.read_text(encoding="utf-8", errors="ignore"))

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
    return "html", _clean_html(response.text)


def _first_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _last_float_row(patterns: list[str], text: str) -> float | None:
    """Extract the final numeric period from a 3-column financial table row."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            groups = [g for g in match.groups() if g is not None]
            if groups:
                return float(groups[-1].replace(",", ""))
    return None


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
    _, report_text = _download(
        meta["financial_report_url"], ".html", f"financial_{quarter}"
    )

    # TCS press releases use slightly different labels across quarters, so the
    # extractor intentionally has multiple aliases and a financial-table fallback.
    revenue_usd = _first_float(
        [
            r"Revenue(?: at)?\s+US\$\s*([\d,]+)\s*million",
            r"Revenue(?: at)?\s*\$\s*([\d,]+)\s*(?:Mn|million)",
            r"4QFY26 Revenue\s+\$\s*([\d,]+)\s*Mn",
            r"Revenue\s*\|\s*[\d,]+\s*\|\s*([\d,]+)",
        ],
        report_text,
    )

    revenue_inr_cr = _first_float(
        [
            r"Revenue\s+at\s+₹\s*([\d,]+)\s*crore",
            r"INR Revenue\s+of\s+₹\s*([\d,]+)\s*Mn",
        ],
        report_text,
    )

    # Use the quarterly IFRS table to avoid accidentally picking the annual net income.
    quarterly_table_start = report_text.lower().find("for the three-month")
    quarterly_table = report_text[quarterly_table_start : quarterly_table_start + 6500] if quarterly_table_start >= 0 else report_text

    if revenue_usd is None:
        revenue_usd = _last_float_row(
            [r"Revenue\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)"],
            quarterly_table,
        )

    net_profit_usd_mn = _first_float(
        [
            r"Net (?:Income|income)\s+at\s+US\$\s*([\d,]+)\s*million",
            r"Net income\s+\$\s*([\d,]+)\s*(?:Mn|million)",
        ],
        report_text,
    )
    if net_profit_usd_mn is None:
        net_profit_usd_mn = _last_float_row(
            [r"Net income\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)"],
            quarterly_table,
        )

    operating_margin = _first_float(
        [r"Operating Margin(?: at|:)\s*([\d.]+)%"], report_text
    )
    net_margin = _first_float(
        [r"Net Margin(?: at|:)\s*([\d.]+)%"], report_text
    )
    tcv = _first_float(
        [
            r"TCV[^$]{0,100}US\$\s*([\d.]+)\s*billion",
            r"TCV[^$]{0,100}\$\s*([\d.]+)\s*billion",
        ],
        report_text,
    )
    ai_rev = _first_float(
        [
            r"Annualized AI (?:Services )?Revenue[^$]{0,30}(?:US\$|\$)\s*([\d.]+)\s*billion",
            r"AI (?:Services )?Revenue[^$]{0,60}(?:US\$|\$)\s*([\d.]+)\s*billion",
        ],
        report_text,
    )
    workforce = _first_float(
        [r"Workforce strength:\s*([\d,]+)", r"Employee Headcount:\s*([\d,]+)"],
        report_text,
    )
    attrition = _first_float(
        [r"Attrition(?:\s*\([^)]*\))?[^\d]{0,30}([\d.]+)%"],
        report_text,
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
