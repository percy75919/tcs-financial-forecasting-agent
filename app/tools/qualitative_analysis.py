import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf as fitz
import httpx
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from ..prompts import QUALITATIVE_SYNTHESIS_PROMPT

settings = get_settings()
MANIFEST_PATH = settings.data_dir / "source_manifest.json"


class FastEmbedEmbeddings:
    """Small LangChain-compatible wrapper around Qdrant FastEmbed.

    Uses ONNX Runtime rather than PyTorch, which avoids the Windows torch.dll
    application-control issue and keeps embedding generation local.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _read_transcript(quarter: str) -> tuple[str, str]:
    url = _load_manifest()["quarters"][quarter]["transcript_url"]
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"transcript_{quarter}.pdf"
    if path.exists() and path.stat().st_size > 100:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc), url

    headers = {"User-Agent": "Mozilla/5.0 TCS-Forecasting-Agent/1.0"}
    with httpx.Client(timeout=settings.request_timeout_seconds, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc), url


def _build_index(quarters: list[str]) -> Chroma:
    embeddings = FastEmbedEmbeddings(settings.local_embedding_model)
    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

    existing = vector_store.get(include=["metadatas"])
    existing_quarters = {m.get("quarter") for m in existing.get("metadatas", []) if m}
    docs: list[Document] = []
    for quarter in quarters:
        if quarter in existing_quarters:
            continue
        text, url = _read_transcript(quarter)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "quarter": quarter,
                    "source": url,
                    "chunk_id": hashlib.sha1(f"{quarter}-{i}".encode()).hexdigest(),
                },
            ))
    if docs:
        vector_store.add_documents(docs)
    return vector_store


def _synthesize(passages: list[Document]) -> dict[str, Any]:
    llm = ChatOllama(model=settings.ollama_model, temperature=0, base_url=settings.ollama_base_url)
    context = "\n\n".join(
        f"[{d.metadata['quarter']}] {d.page_content}\nSOURCE: {d.metadata['source']}"
        for d in passages
    )
    prompt = f"""{QUALITATIVE_SYNTHESIS_PROMPT}\n\nTRANSCRIPT EVIDENCE:\n{context}\n"""
    response = llm.invoke(prompt)
    return {
        "summary": response.content,
        "evidence": [
            {"quarter": d.metadata["quarter"], "source": d.metadata["source"], "snippet": d.page_content}
            for d in passages
        ],
    }


@tool("QualitativeAnalysisTool")
def qualitative_analysis_tool(quarters: list[str], query: str) -> str:
    """Run semantic RAG over 2-3 TCS earnings call transcripts and synthesize themes, sentiment, risks and opportunities."""
    vector_store = _build_index(quarters)
    search_queries = [
        query,
        "demand environment macro geopolitical uncertainty project deferrals client spending outlook",
        "AI services monetization agentic AI transformation deal wins partnerships opportunity",
        "risks margin pressure wage hikes profitability uncertainty sector weakness",
    ]
    passages: list[Document] = []
    seen = set()
    for q in search_queries:
        for doc in vector_store.similarity_search(q, k=max(2, settings.top_k_transcript_chunks // 2)):
            cid = doc.metadata.get("chunk_id")
            if cid not in seen:
                passages.append(doc)
                seen.add(cid)
    synthesis = _synthesize(passages[: settings.top_k_transcript_chunks])
    return json.dumps({"tool": "QualitativeAnalysisTool", **synthesis}, indent=2)
