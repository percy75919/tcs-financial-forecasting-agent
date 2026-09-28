from typing import Literal

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    task: str = Field(
        default=(
            "Analyze the financial reports and earnings call transcripts for the last three "
            "reported TCS quarters and provide a qualitative forecast for the upcoming quarter. "
            "Identify key financial trends, management outlook, major risks and opportunities."
        ),
        min_length=20,
    )
    quarters: list[str] = Field(
        default_factory=lambda: ["Q1_FY27", "Q4_FY26", "Q3_FY26"],
        min_length=2,
        max_length=3,
    )


class MetricEvidence(BaseModel):
    metric: str
    value: float | None = None
    unit: str = ""
    period: str
    source: str
    evidence: str = ""


class FinancialQuarter(BaseModel):
    quarter: str
    revenue_usd_mn: float | None = None
    revenue_inr_cr: float | None = None
    net_profit_usd_mn: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    tcv_usd_bn: float | None = None
    ai_annualized_revenue_usd_bn: float | None = None
    workforce: int | None = None
    attrition_pct: float | None = None
    evidence: list[MetricEvidence] = Field(default_factory=list)


class TrendFinding(BaseModel):
    theme: str
    direction: Literal["improving", "stable", "weakening", "mixed"]
    evidence: str
    source_quarters: list[str]


class QualitativeAnalysis(BaseModel):
    recurring_themes: list[str] = Field(default_factory=list)
    management_sentiment: Literal["constructive", "mixed", "cautious", "unclear"] = "unclear"
    forward_looking_statements: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    trends: list[TrendFinding] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)


class SourceTrace(BaseModel):
    quarter: str
    source_type: Literal["financial_report", "earnings_transcript"]
    source: str
    claim: str


class ForecastResponse(BaseModel):
    company: str = "Tata Consultancy Services"
    forecast_period: str
    outlook: Literal["positive", "constructive", "mixed", "cautious"]
    confidence: Literal["high", "medium", "low"]
    executive_summary: str
    financial_trends: list[str]
    management_outlook: list[str]
    key_risks: list[str]
    key_opportunities: list[str]
    forecast_rationale: list[str]
    supporting_quarters: list[FinancialQuarter]
    source_trace: list[SourceTrace]
    disclaimer: str = (
        "This is a qualitative, evidence-grounded business outlook, not investment advice or a point estimate. "
        "TCS does not provide specific revenue or earnings guidance."
    )


class HealthResponse(BaseModel):
    status: str
    service: str
