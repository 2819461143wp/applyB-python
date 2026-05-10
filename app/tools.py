from __future__ import annotations

from app.knowledge_base import KnowledgeBase

REQUIRED_MATERIALS = ["申请书", "身份证", "营业执照"]

kb = KnowledgeBase()


def knowledge_search(query: str, top_k: int = 3) -> dict:
    results = kb.search(query, top_k)
    return {"results": results}


def check_completeness(materials: list[str]) -> dict:
    missing = [item for item in REQUIRED_MATERIALS if item not in materials]
    return {
        "required": REQUIRED_MATERIALS,
        "missing": missing,
        "complete": len(missing) == 0,
    }
