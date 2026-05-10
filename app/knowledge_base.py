from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter


class KnowledgeBase:
    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name="water_rules")
        self._bootstrapped = False

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        if self.collection.count() > 0:
            self._bootstrapped = True
            return

        rules = [
            "申请材料需包含申请书、身份证、营业执照。",
            "申请书中申请人姓名、证件号、联系方式为必填字段。",
            "申请书中取水地点、申请期限属于关键信息，不得空缺。",
            "取水用途选择其他时，必须补充具体用途说明。",
            "证照必须在有效期内，过期证件不予通过。",
            "申请表中的身份信息应与附件证照信息一致。",
            "申请表项目名称、法定代表人与附件报告或营业执照信息应保持一致。",
            "申请取水量不应高于水资源论证报告测算用水量。",
            "取水口位于饮用水水源保护区、自然保护区等区域时，应按法规限制或禁止取水。",
            "水资源论证报告需说明第三方影响并给出补救措施。",
            "申请取水许可期限不得超过建设项目批准期限。",
            "水资源论证报告超过5年应重新评估有效性。",
            "申请依据的法律法规或标准应使用现行有效版本。",
            "申请材料应包含权属证明或合法租赁协议等支撑文件。",
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
        chunks: list[str] = []
        for text in rules:
            chunks.extend(splitter.split_text(text))

        ids = [f"seed-{index}" for index in range(len(chunks))]
        vectors = [self._embed(chunk) for chunk in chunks]
        metadatas = [{"source": "seed_rules"} for _ in chunks]
        self.collection.add(documents=chunks, embeddings=vectors, ids=ids, metadatas=metadatas)
        self._bootstrapped = True

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        self.bootstrap()
        result = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        items: list[dict[str, Any]] = []
        for idx, content in enumerate(docs):
            source = metas[idx].get("source", "unknown") if idx < len(metas) and metas[idx] else "unknown"
            distance = dists[idx] if idx < len(dists) else 0.0
            items.append(
                {
                    "content": content,
                    "source": source,
                    "score": float(1 / (1 + max(distance, 0))),
                }
            )
        return items

    def _embed(self, text: str) -> list[float]:
        # Lightweight deterministic embedding to keep local setup runnable without external models.
        base = [0.0] * 64
        for i, b in enumerate(text.encode("utf-8")):
            base[i % 64] += (b % 23) / 23.0
        norm = sum(v * v for v in base) ** 0.5 or 1.0
        return [v / norm for v in base]
