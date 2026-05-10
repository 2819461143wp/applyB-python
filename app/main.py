from __future__ import annotations

from fastapi import FastAPI

from app.agent import agent, extract_draft
from app.schemas import CompletenessCheckRequest, ExtractionDraft, KnowledgeQuery, PrecheckRequest
from app.tools import check_completeness, knowledge_search

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
