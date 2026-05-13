from __future__ import annotations

import json
import re
import tempfile
import uuid
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
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

    def ingest_file(self, filename: str, content: bytes) -> dict[str, Any]:
        self.bootstrap()
        text, parser = self._parse_upload(filename, content)
        if not text.strip():
            return {
                "filename": filename,
                "parser": parser,
                "chunksAdded": 0,
                "totalVectors": self.collection.count(),
                "preview": "",
                "contentText": "",
                "contentLength": 0,
                "message": "未能从文档中解析出可入库文本。",
            }

        parsed_text = text.strip()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=140,
            chunk_overlap=24,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        chunks = [chunk.strip() for chunk in splitter.split_text(parsed_text) if chunk.strip()]
        doc_id = uuid.uuid4().hex[:12]
        ids = [f"upload-{doc_id}-{index}" for index in range(len(chunks))]
        vectors = [self._embed(chunk) for chunk in chunks]
        metadatas = [
            {
                "source": filename,
                "parser": parser,
                "doc_id": doc_id,
                "chunk": index,
            }
            for index in range(len(chunks))
        ]
        self.collection.add(documents=chunks, embeddings=vectors, ids=ids, metadatas=metadatas)
        return {
            "filename": filename,
            "docId": doc_id,
            "parser": parser,
            "chunksAdded": len(chunks),
            "totalVectors": self.collection.count(),
            "preview": parsed_text[:500],
            "contentText": parsed_text,
            "contentLength": len(parsed_text),
            "message": "知识库文档已解析并写入向量库。",
        }

    def list_documents(self) -> dict[str, Any]:
        self.bootstrap()
        documents = self._document_groups()
        return {
            "documents": documents,
            "totalDocuments": len(documents),
            "totalVectors": self.collection.count(),
        }

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        self.bootstrap()
        documents = self._document_groups()
        target = next((item for item in documents if item["docId"] == doc_id), None)
        if target is None:
            return {
                "deleted": False,
                "docId": doc_id,
                "deletedChunks": 0,
                "totalVectors": self.collection.count(),
                "message": "未找到该知识库文档。",
                "documents": documents,
            }
        if not target["deletable"]:
            return {
                "deleted": False,
                "docId": doc_id,
                "deletedChunks": 0,
                "totalVectors": self.collection.count(),
                "message": "内置课堂规则不可删除。",
                "documents": documents,
            }

        before = self.collection.count()
        self.collection.delete(where={"doc_id": doc_id})
        after = self.collection.count()
        deleted_chunks = max(0, before - after)
        return {
            "deleted": deleted_chunks > 0,
            "docId": doc_id,
            "filename": target["filename"],
            "deletedChunks": deleted_chunks,
            "totalVectors": after,
            "message": f"已删除 {target['filename']} 的 {deleted_chunks} 个向量切片。",
            "documents": self._document_groups(),
        }

    def vector_points(self, limit: int = 160, similarity_threshold: float = 0.65) -> dict[str, Any]:
        self.bootstrap()
        data = self.collection.get(
            limit=max(1, min(limit, 300)),
            include=["documents", "metadatas", "embeddings"],
        )
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])
        ids = data.get("ids", [])
        coords = self._project_embeddings([[float(value) for value in embedding] for embedding in embeddings])
        points: list[dict[str, Any]] = []
        for index, embedding in enumerate(embeddings):
            vector = [float(value) for value in embedding]
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            content = documents[index] if index < len(documents) else ""
            x, y = coords[index] if index < len(coords) else (0.0, 0.0)
            points.append(
                {
                    "id": ids[index] if index < len(ids) else f"point-{index}",
                    "x": x,
                    "y": y,
                    "source": metadata.get("source", "unknown"),
                    "parser": metadata.get("parser", "seed"),
                    "docId": metadata.get("doc_id", metadata.get("source", "seed")),
                    "chunk": int(metadata.get("chunk", index)),
                    "content": content[:180],
                }
            )
        edges = self._similarity_edges(points, [[float(value) for value in embedding] for embedding in embeddings], similarity_threshold)
        sources = sorted({point["source"] for point in points})
        return {
            "total": self.collection.count(),
            "points": points,
            "edges": edges,
            "sources": sources,
            "similarityThreshold": similarity_threshold,
        }

    def _document_groups(self) -> list[dict[str, Any]]:
        total = self.collection.count()
        if total <= 0:
            return []

        data = self.collection.get(
            limit=max(1, total),
            include=["documents", "metadatas"],
        )
        contents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        grouped: dict[str, dict[str, Any]] = {}
        for index, metadata in enumerate(metadatas):
            metadata = metadata or {}
            source = str(metadata.get("source") or "unknown")
            doc_id = str(metadata.get("doc_id") or f"seed:{source}")
            parser = str(metadata.get("parser") or "seed")
            document = grouped.setdefault(
                doc_id,
                {
                    "docId": doc_id,
                    "filename": source,
                    "source": source,
                    "parser": parser,
                    "chunks": 0,
                    "preview": "",
                    "deletable": "doc_id" in metadata,
                },
            )
            document["chunks"] += 1
            if parser != "seed":
                document["parser"] = parser
            if not document["preview"] and index < len(contents):
                document["preview"] = str(contents[index])[:180]

        return sorted(
            grouped.values(),
            key=lambda item: (not item["deletable"], item["filename"], item["docId"]),
        )

    def _embed(self, text: str) -> list[float]:
        # Lightweight deterministic embedding to keep local setup runnable without external models.
        base = [0.0] * 64
        for i, b in enumerate(text.encode("utf-8")):
            base[i % 64] += (b % 23) / 23.0
        norm = sum(v * v for v in base) ** 0.5 or 1.0
        return [v / norm for v in base]

    def _project_embeddings(self, embeddings: list[list[float]]) -> list[tuple[float, float]]:
        if not embeddings:
            return []
        if len(embeddings) == 1:
            return [(0.0, 0.0)]
        try:
            import numpy as np

            matrix = np.array(embeddings, dtype=float)
            matrix = matrix - matrix.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(matrix, full_matrices=False)
            components = vh[:2].T
            projected = matrix @ components
            if projected.shape[1] == 1:
                projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
            return [(float(row[0]), float(row[1])) for row in projected]
        except Exception:
            return [
                (
                    sum(vector[0::2]) / max(1, len(vector[0::2])),
                    sum(vector[1::2]) / max(1, len(vector[1::2])),
                )
                for vector in embeddings
            ]

    def _similarity_edges(
        self,
        points: list[dict[str, Any]],
        embeddings: list[list[float]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for i, source_point in enumerate(points):
            candidates: list[tuple[float, int]] = []
            for j in range(i + 1, len(points)):
                target_point = points[j]
                if source_point["source"] != target_point["source"]:
                    continue
                score = self._cosine(embeddings[i], embeddings[j])
                if score >= threshold:
                    candidates.append((score, j))
            for score, j in sorted(candidates, reverse=True)[:6]:
                edges.append(
                    {
                        "from": source_point["id"],
                        "to": points[j]["id"],
                        "source": source_point["source"],
                        "similarity": round(score, 4),
                    }
                )
        return edges

    def _cosine(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5 or 1.0
        right_norm = sum(b * b for b in right) ** 0.5 or 1.0
        return dot / (left_norm * right_norm)

    def _parse_upload(self, filename: str, content: bytes) -> tuple[str, str]:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            return self._decode_plain(content), "plain"
        if suffix == ".docx":
            return self._parse_docx(content), "docx-xml"

        mineru_text = self._parse_with_mineru(filename, content)
        if mineru_text.strip():
            return mineru_text, "mineru"

        if suffix == ".pdf":
            return self._parse_pdf(content), "pdf-fallback"
        return self._decode_plain(content), "plain-fallback"

    def _parse_with_mineru(self, filename: str, content: bytes) -> str:
        try:
            from mineru.cli.common import do_parse
        except Exception:
            return ""

        safe_stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fa5]+", "_", Path(filename).stem).strip("_") or "document"
        try:
            with tempfile.TemporaryDirectory(prefix="water-mineru-") as tmp:
                output_dir = Path(tmp)
                do_parse(
                    output_dir=str(output_dir),
                    pdf_file_names=[safe_stem],
                    pdf_bytes_list=[content],
                    p_lang_list=["ch"],
                    backend="pipeline",
                    parse_method="auto",
                    formula_enable=True,
                    table_enable=True,
                    f_dump_md=True,
                    f_dump_content_list=True,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_orig_pdf=False,
                    start_page_id=0,
                    end_page_id=None,
                )
                markdown_files = list(output_dir.rglob("*.md"))
                if markdown_files:
                    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in markdown_files)
                content_files = list(output_dir.rglob("*content_list*.json"))
                if content_files:
                    items = json.loads(content_files[0].read_text(encoding="utf-8", errors="ignore"))
                    return "\n".join(str(item.get("text") or item.get("html") or "") for item in items if isinstance(item, dict))
        except Exception:
            return ""
        return ""

    def _decode_plain(self, content: bytes) -> str:
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return ""

    def _parse_docx(self, content: bytes) -> str:
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                document_xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        except Exception:
            return ""
        try:
            root = ET.fromstring(document_xml)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            lines: list[str] = []
            for paragraph in root.findall(".//w:p", namespace):
                pieces = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
                line = "".join(pieces).strip()
                if line:
                    lines.append(line)
            if lines:
                return "\n".join(lines).strip()
        except ET.ParseError:
            pass

        text = re.sub(r"<[^>]+>", "\n", document_xml)
        return re.sub(r"\n+", "\n", text).strip()

    def _parse_pdf(self, content: bytes) -> str:
        raw = content.decode("latin-1", errors="ignore")
        matches = re.findall(r"\(([^()]*)\)", raw)
        decoded = [
            match.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore").strip()
            for match in matches
        ]
        return " ".join(item for item in decoded if item).strip()
