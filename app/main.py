from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form

from app.agent import agent, extract_draft
from app.schemas import CompletenessCheckRequest, ExtractionDraft, KnowledgeQuery, PrecheckRequest
from app.tools import check_completeness, delete_knowledge_document, ingest_knowledge_file, knowledge_documents, knowledge_search, vector_visualization

app = FastAPI(title="Water Approval AI Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ai/precheck")
def precheck(payload: PrecheckRequest) -> dict:
    return agent.run(payload).model_dump()


@app.post("/api/ai/extract", response_model=ExtractionDraft)
def extract(payload: PrecheckRequest) -> ExtractionDraft:
    return extract_draft(payload)


@app.get("/api/mcp/tools")
def mcp_tools() -> dict:
    return {
        "tools": [
            {
                "name": "knowledge_search",
                "description": "接收查询语句，返回相关法规片段。",
                "input_schema": {"query": "string", "top_k": "int"},
            },
            {
                "name": "check_completeness",
                "description": "接收申请材料列表，对照清单返回缺失项。",
                "input_schema": {"materials": "list[string]"},
            },
        ]
    }


@app.post("/api/mcp/tools/knowledge_search")
def mcp_knowledge_search(query: KnowledgeQuery) -> dict:
    return knowledge_search(query.query, query.top_k)


@app.post("/api/mcp/tools/check_completeness")
def mcp_check_completeness(payload: CompletenessCheckRequest) -> dict:
    return check_completeness(payload.materials)


@app.post("/api/ai/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...), filename: str | None = Form(default=None)) -> dict:
    content = await file.read()
    display_name = (filename or file.filename or "knowledge-document").strip()
    return ingest_knowledge_file(display_name, content)


@app.get("/api/ai/knowledge/documents")
def list_knowledge_documents() -> dict:
    return knowledge_documents()


@app.delete("/api/ai/knowledge/documents/{doc_id}")
def remove_knowledge_document(doc_id: str) -> dict:
    return delete_knowledge_document(doc_id)


@app.get("/api/ai/knowledge/vectors")
def knowledge_vectors(limit: int = 120) -> dict:
    return vector_visualization(limit)
