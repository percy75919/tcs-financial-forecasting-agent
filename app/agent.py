import json
from typing import Any

from langchain_ollama import ChatOllama

from .config import get_settings
from .prompts import MASTER_SYSTEM_PROMPT, USER_TASK_TEMPLATE
from .schemas import ForecastResponse
from .tools.financial_data_extractor import financial_data_extractor_tool
from .tools.qualitative_analysis import qualitative_analysis_tool

settings = get_settings()


def build_agent():
    """Build the final local structured-output model.

    Tool execution is orchestrated deterministically before synthesis so both
    required specialist tools are always used. The final model call is then
    constrained to the ForecastResponse schema.
    """
    llm = ChatOllama(
        model=settings.ollama_model,
        temperature=0,
        base_url=settings.ollama_base_url,
    )
    return llm.with_structured_output(ForecastResponse, method="json_schema")


def _run_tools(quarters: list[str], task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run both required specialist tools and return parsed evidence."""
    financial_raw = financial_data_extractor_tool.invoke({"quarters": quarters})
    qualitative_raw = qualitative_analysis_tool.invoke({
        "quarters": quarters,
        "query": task,
    })

    return json.loads(financial_raw), json.loads(qualitative_raw)


def generate_forecast(task: str, quarters: list[str]) -> ForecastResponse:
    financial_evidence, qualitative_evidence = _run_tools(quarters, task)

    prompt = USER_TASK_TEMPLATE.format(
        quarters=", ".join(quarters),
        task=task,
    )

    synthesis_prompt = f"""{MASTER_SYSTEM_PROMPT}

{prompt}

SPECIALIST TOOL EVIDENCE
========================
FinancialDataExtractorTool:
{json.dumps(financial_evidence, indent=2)}

QualitativeAnalysisTool:
{json.dumps(qualitative_evidence, indent=2)}

SYNTHESIS REQUIREMENTS
======================
- Return ONLY data conforming to the ForecastResponse schema.
- Do not invent reported metrics or management statements.
- Every material rationale must be traceable to the supplied evidence.
- Use the three supplied reported quarters to assess the upcoming quarter qualitatively.
- Do not give a point-estimate revenue or earnings forecast.
- Do not provide investment advice.
"""

    structured_llm = build_agent()
    result = structured_llm.invoke(synthesis_prompt)
    if isinstance(result, ForecastResponse):
        return result
    if isinstance(result, dict):
        return ForecastResponse.model_validate(result)
    raise RuntimeError("Structured LLM did not return a ForecastResponse")
