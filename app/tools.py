from __future__ import annotations

from app.knowledge_base import KnowledgeBase

REQUIRED_MATERIALS = ["申请书", "身份证", "营业执照"]

kb = KnowledgeBase()


def knowledge_search(query: str, top_k: int = 3) -> dict:
    results = kb.search(query, top_k)
    return {"results": results}


def ingest_knowledge_file(filename: str, content: bytes) -> dict:
    return kb.ingest_file(filename, content)


def knowledge_documents() -> dict:
    return kb.list_documents()


def delete_knowledge_document(doc_id: str) -> dict:
    return kb.delete_document(doc_id)


def vector_visualization(limit: int = 120) -> dict:
    return kb.vector_points(limit)


def check_completeness(materials: list[str]) -> dict:
    missing = [item for item in REQUIRED_MATERIALS if item not in materials]
    return {
        "required": REQUIRED_MATERIALS,
        "missing": missing,
        "complete": len(missing) == 0,
    }
