# TCS Financial Forecasting Agent

A FastAPI + LangChain application that researches Tata Consultancy Services (TCS) across the latest three reported quarters, extracts financial metrics, performs RAG over earnings-call transcripts, and returns a structured qualitative outlook for the upcoming quarter.

## 1. Project overview

The application follows a two-tool agent architecture:

```text
Client
  |
  v
POST /forecast
  |
  +--> MySQL request log
  |
  v
LangChain agent
  |
  +--> FinancialDataExtractorTool
  |      |- downloads official TCS quarterly result pages
  |      |- parses reported financial statements / highlights
  |      |- returns metrics + evidence + source URLs
  |
  +--> QualitativeAnalysisTool
         |- downloads 2-3 official TCS earnings-call transcripts
         |- splits transcript text into chunks
         |- embeds chunks with a local FastEmbed/ONNX model
         |- stores/searches them in Chroma
         |- synthesizes recurring themes, risks, opportunities and management outlook
  |
  v
Structured Pydantic ForecastResponse
  |
  v
MySQL final-output log
```

The design intentionally separates deterministic financial extraction from semantic transcript analysis. This reduces the chance that the LLM invents a number that should have been read from a filing. The application can run fully locally: Ollama provides the final chat model, while FastEmbed provides local transcript embeddings.

The agent is asked to use both tools before returning a result. LangChain's current `create_agent` factory supports a tool-calling loop and structured response output; the project uses a Pydantic response schema so the HTTP response is machine-readable. See the current LangChain reference for `create_agent` and structured responses.

## 2. Source documents

The default snapshot is based on:

- Q1 FY27, quarter ended June 30, 2026
- Q4 FY26, quarter ended March 31, 2026
- Q3 FY26, quarter ended December 31, 2025

The source registry is in `data/source_manifest.json`. URLs point to TCS investor-relations materials. The application downloads them on first use and caches them under `data/raw/`.

The source selection is not hard-coded into the prompt; it is configuration. This makes it straightforward to update the quarter set after a new TCS earnings release.

## 3. Agent and tool design

### FinancialDataExtractorTool

Purpose: extract reported quarterly metrics and provide provenance.

It currently looks for:

- Revenue in USD and/or INR
- Net profit
- Operating margin
- Net margin
- Total Contract Value (TCV)
- Annualized AI revenue
- Workforce
- Attrition

The tool uses HTML parsing for TCS press-release pages and PyMuPDF for PDF material when needed. Extraction is regex-driven with multiple aliases and returns a short evidence window around each match. That evidence is kept in the structured output so the final synthesis can be audited.

### QualitativeAnalysisTool

Purpose: run semantic retrieval over earnings-call transcripts and synthesize management commentary.

Pipeline:

1. Download official transcripts as PDFs.
2. Extract text with PyMuPDF.
3. Split into overlapping chunks.
4. Embed with the local Sentence-Transformers model `sentence-transformers/all-MiniLM-L6-v2`.
5. Persist vectors in a local Chroma collection.
6. Run multiple semantic queries covering demand, AI monetization, risks and margin pressure.
7. Send the retrieved evidence to an LLM subagent for a compact qualitative synthesis.

The RAG layer returns the quarter and source URL for each retrieved chunk. The FastEmbed model is downloaded once and then reused from its local cache.

### Master prompt

The master prompt lives in `app/prompts.py`. Its core controls are:

- mandatory use of both tools;
- source-grounded claims only;
- explicit separation of historical facts and management statements from inference;
- no fabricated numerical forecast;
- lower confidence when evidence is missing or conflicting;
- source traceability;
- no private chain-of-thought in the response.

The service returns concise forecast rationales rather than hidden reasoning transcripts.

## 4. AI stack and reasoning approach

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Small, typed HTTP service |
| Agent orchestration | LangChain `create_agent` | Tool-calling loop + structured output |
| LLM | Ollama `ChatOllama` via `langchain-ollama` | Local tool calling and structured generation without API credits |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (local, ONNX Runtime) | Avoids OpenAI embedding charges and keeps transcript text local during embedding |
| Vector DB | Chroma via `langchain-chroma` | Local persistent RAG store |
| Document parsing | PyMuPDF + BeautifulSoup | PDF and HTML support without OCR when text is available |
| Validation | Pydantic | Predictable JSON contract |
| Persistence | MySQL 8.0 + SQLAlchemy | Request and final-output audit log |

**Important billing note:** OpenAI embeddings are not used. The RAG index is built with a local FastEmbed model running through ONNX Runtime, and the final agent/tool synthesis uses an Ollama model running locally. This removes the need for OpenAI API credits for normal development and demo runs.

The forecast is qualitative. The agent does not manufacture a numeric next-quarter EPS or revenue target. It assesses directional momentum from the observed sequence of financial metrics plus management commentary.

## 5. What the AI achieves end-to-end

For the default three-quarter window, the system can retrieve:

- TCS financial result pages for Q1 FY27, Q4 FY26 and Q3 FY26;
- the corresponding earnings-call transcripts;
- sequential revenue and margin measures;
- order-book/TCV and AI-revenue signals where disclosed;
- recurring transcript themes such as demand, client deferrals, AI transformation, pricing/margin pressure, macro uncertainty and management expectations.

The final response is a `ForecastResponse` JSON object containing:

- outlook direction;
- confidence;
- executive summary;
- financial trends;
- management outlook;
- risks and opportunities;
- forecast rationale;
- supporting quarter metrics;
- source trace.

## 6. Guardrails and evaluation

### Guardrails

- Official TCS investor-relations URLs are the primary data source.
- Tool outputs include source URLs and evidence snippets.
- Pydantic enforces the final response schema.
- The LLM temperature is set to 0 for the synthesis path.
- A single bounded retry is used on agent failures.
- The prompt prohibits invented numbers and distinguishes management statements from independent facts.
- Missing or conflicting evidence is expected to reduce confidence rather than be silently filled in.

### Basic evaluation

Run:

```bash
pytest -q
```

The included tests validate regex extraction behavior, the structured response contract and the API health endpoint. For a stronger submission, add snapshot tests against a frozen copy of the TCS source documents and a small human-annotated rubric for theme extraction.

## 7. Setup

### Prerequisites

- Python 3.10+
- Docker Desktop / Docker Engine
- Ollama for Windows/macOS/Linux
- MySQL 8.0 (Docker Compose is provided)

#### Install and start Ollama

Download Ollama from the official installer and install it. On Windows, the official download page provides a Windows installer. The Ollama service listens locally on `http://localhost:11434` by default.

Then pull the default model used by this project:

```bash
ollama pull qwen3:8b
```

Verify it is available:

```bash
ollama list
```

The code uses LangChain's `ChatOllama` integration to call this local model.

### Step 1: clone the repository

```bash
git clone <your-github-repository-url>
cd tcs-forecasting-agent
```

### Step 2: create a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: configure environment variables

```bash
Copy-Item .env.example .env
```

PowerShell/macOS/Linux alternatives are fine; then edit `.env` if you want to change the Ollama model or base URL.

The default MySQL values match the included Docker Compose file. Local embeddings are configured by `LOCAL_EMBEDDING_MODEL`, and the local chat model is configured by `OLLAMA_MODEL`; no OpenAI API key or OpenAI embedding credit is required.

### Step 5: start MySQL

```bash
docker compose up -d mysql
```

### Step 6: pre-ingest documents (recommended)

```bash
python -m scripts.ingest
```

This downloads the three configured financial result pages and transcripts into `data/raw/` and builds the Chroma index.

If the network is unavailable during ingestion, the service will fail with a source-fetch error rather than silently fabricating data.

## 8. Run the FastAPI service

```bash
uvicorn app.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Forecast request:

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter. Identify revenue and margin trends, summarize management outlook, and highlight significant risks and opportunities.",
    "quarters": ["Q1_FY27", "Q4_FY26", "Q3_FY26"]
  }'
```

PowerShell example:

```powershell
$body = @{
  task = "Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter. Identify revenue and margin trends, summarize management outlook, and highlight significant risks and opportunities."
  quarters = @("Q1_FY27", "Q4_FY26", "Q3_FY26")
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/forecast -ContentType "application/json" -Body $body
```

Example Forecast Response
A successful `/forecast` request returns structured JSON containing the forecast, supporting financial trends, management outlook, risks, opportunities, rationale, and source traceability.

```json
{
  "company": "Tata Consultancy Services",
  "forecast_period": "string",
  "outlook": "positive",
  "confidence": "high",
  "executive_summary": "string",
  "financial_trends": [
    "string"
  ],
  "management_outlook": [
    "string"
  ],
  "key_risks": [
    "string"
  ],
  "key_opportunities": [
    "string"
  ],
  "forecast_rationale": [
    "string"
  ],
  "supporting_quarters": [
    {
      "quarter": "string",
      "revenue_usd_mn": 0,
      "revenue_inr_cr": 0,
      "net_profit_usd_mn": 0,
      "operating_margin_pct": 0,
      "net_margin_pct": 0,
      "tcv_usd_bn": 0,
      "ai_annualized_revenue_usd_bn": 0,
      "workforce": 0,
      "attrition_pct": 0,
      "evidence": [
        {
          "metric": "string",
          "value": 0,
         "unit": "",
          "period": "string",
          "source": "string",
          "evidence": ""
        }
      ]
    }
  ],
  "source_trace": [
    {
      "quarter": "string",
      "source_type": "financial_report",
      "source": "string",
      "claim": "string"
    }
  ],
  "disclaimer": "This is a qualitative, evidence-grounded business outlook, not investment advice or a point estimate. TCS does not provide specific revenue or earnings guidance."
}
```

## 9. MySQL logging

Every `/forecast` request creates a row in `forecast_logs` containing:

- request timestamp;
- task text;
- requested quarters;
- status;
- final JSON output, when successful;
- error text, when unsuccessful.

The table is created automatically on application startup.

## 10. Production considerations

For a production deployment, add:

- a secret manager rather than `.env` for credentials;
- connection pooling settings tuned for workload;
- authenticated API access and rate limits;
- document checksum/version tracking;
- scheduled source refresh after earnings releases;
- OCR fallback for scanned PDFs if a source ever arrives without a text layer;
- a real evaluation set with expected financial metrics and annotated transcript themes;
- observability for latency, token usage, tool-call failures and retrieval quality.

## 11. Tradeoffs and limitations

1. **Qualitative, not a precise numeric earnings model.** This matches the requirement for a reasoned business outlook but is not a substitute for a full bottoms-up financial model.
2. **Official-source availability.** The service depends on TCS pages/PDFs being reachable. The design fails loudly on missing source data to avoid hallucination.
3. **Embedding cost vs. retrieval quality.** The project uses a local Sentence-Transformers model to avoid OpenAI embedding charges. The tradeoff is a larger local dependency/model download and CPU inference during ingestion/querying.
4. **Chroma is intentionally local.** This makes the assignment easy to run but is not a distributed production vector database.
5. **No OCR by default.** Most TCS investor documents expose a text layer. OCR should be added only for image-only documents.
6. **Management outlook is not a formal earnings forecast.** TCS explicitly states that it does not provide specific revenue or earnings guidance; the agent therefore frames management commentary as directional evidence.

## 12. Current snapshot verified for this assignment

As of September 28, 2026, TCS's investor-relations calendar lists July 9, 2026 for Q1 FY27 earnings, April 9, 2026 for Q4 FY26, and January 12, 2026 for Q3 FY26. The TCS investor-relations page exposes the Q1 FY27 transcript, while the Q4 and Q3 official transcript PDFs are also available from TCS.

The Q1 FY27 published results reported revenue of US$7.624 billion, 24.0% operating margin, 19.2% net margin, US$9.5 billion TCV, and US$2.6 billion annualized AI revenue. The Q1 transcript also records management commentary that demand had faced macro/geopolitical uncertainty and that management expected improvement in Q2, while noting wage hikes reduced operating margin sequentially.

Q4 FY26 reported revenue of US$7.621 billion and operating margin of 25.3%, with TCV of US$12.0 billion. Q3 FY26 reported revenue of US$7.509 billion, 25.2% operating margin and US$9.3 billion TCV.

These values are included here only as a verified reference snapshot; the application itself is designed to retrieve the documents dynamically.

## Windows/Python 3.14 troubleshooting

This version intentionally uses FastEmbed with ONNX Runtime for local embeddings instead of Sentence-Transformers/PyTorch. That keeps the RAG embedding path independent of `torch.dll`. FastEmbed currently supports Python 3.14 and selects compatible ONNX Runtime dependencies for that interpreter. If you are using Windows and want to avoid shell activation issues, you can run the virtual environment directly:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.ingest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Do not run the global `pytest` executable if it resolves to a different Python installation. The explicit `.venv\Scripts\python.exe -m ...` commands guarantee that the project's environment is used.


## Local Ollama latency note
The local Qwen3 8B model may take significantly longer on CPU than a hosted API. The default request/download timeout is 180 seconds. The final synthesis uses LangChain structured output with JSON Schema, while the two specialist tools are executed deterministically first so that both required tools are always included in the evidence chain.
