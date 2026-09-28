from app.schemas import ForecastResponse


def test_forecast_schema_is_machine_readable():
    obj = ForecastResponse(
        forecast_period="Q2 FY27",
        outlook="constructive",
        confidence="medium",
        executive_summary="Evidence suggests improving demand with near-term margin pressure.",
        financial_trends=["Revenue momentum improved sequentially."],
        management_outlook=["Management expects demand to improve."],
        key_risks=["Macro uncertainty"],
        key_opportunities=["AI transformation"],
        forecast_rationale=["Sequential growth and deal wins support the view."],
        supporting_quarters=[],
        source_trace=[],
    )
    data = obj.model_dump()
    assert data["company"] == "Tata Consultancy Services"
    assert data["outlook"] in {"positive", "constructive", "mixed", "cautious"}
