"""
Модуль работы с векторным хранилищем ChromaDB.
Загрузка нескольких источников с метаданными, нарезка по статьям (нормативка) и по смыслу (обзоры).
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import time
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError

from openai_client import get_openai_client

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

_STATUTE_BOUNDARY = re.compile(
    r"(?m)^(?=(?:§\s*\d+(?:\.\d+)?[\.\s]|Статья\s+\d+))"
)


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = str(Path(__file__).resolve().parent / "chroma_db")
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана новая коллекция '{collection_name}'")

        self.openai_client = get_openai_client()
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.min_chunk_len = int(os.getenv("RAG_MIN_CHUNK_LEN", "80"))

    def _split_sentences(self, text: str) -> List[str]:
        try:
            from pysbd import Segmenter

            segmenter = Segmenter(language="ru", clean=False)
            return [s.strip() for s in segmenter.segment(text) if s.strip()]
        except Exception:
            return self._split_sentences_regex(text)

    def _split_sentences_regex(self, text: str) -> List[str]:
        parts = re.split(r"([.!?]+\s+)", text)
        full: List[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                full.append((parts[i] + parts[i + 1]).strip())
                i += 2
            else:
                if parts[i].strip():
                    full.append(parts[i].strip())
                i += 1
        return [s for s in full if s]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        overlap_candidate = text[-overlap_size:]
        sentence_starts = [". ", "! ", "? ", "\n"]
        best_start = 0
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        return overlap_candidate.strip()

    def _split_long_block(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        sentences = self._split_sentences(paragraph)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                    overlap_text = self._get_overlap_text(current, overlap)
                    current = (overlap_text + " " + sentence).strip() if overlap_text else sentence
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks

    def _chunk_semantic(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 2 <= chunk_size:
                current = current + "\n\n" + paragraph if current else paragraph
            elif current:
                chunks.append(current)
                overlap_text = self._get_overlap_text(current, overlap)
                current = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    sent_chunks = self._split_long_block(paragraph, chunk_size, overlap)
                    if sent_chunks:
                        chunks.extend(sent_chunks[:-1])
                        current = sent_chunks[-1]
                else:
                    current = paragraph
        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) >= self.min_chunk_len]

    def _split_statute_sections(self, text: str) -> List[str]:
        parts = _STATUTE_BOUNDARY.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1:
            stripped = text.strip()
            return [stripped] if stripped else []
        return parts

    def _section_heading(self, section: str, max_len: int = 160) -> str:
        line = section.strip().split("\n", 1)[0].strip()
        if len(line) > max_len:
            return line[:max_len] + "…"
        return line

    def _kind_label(self, source_kind: str) -> str:
        return {
            "law": "закон РФ",
            "rules": "правила (URDG)",
            "case_law_summary": "обзор судебной практики",
        }.get(source_kind, source_kind)

    def _build_chunks_for_file(
        self,
        text: str,
        source: str,
        source_display: str,
        source_kind: str,
        doc_type: str,
    ) -> List[Tuple[str, Dict[str, str]]]:
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        kind_label = self._kind_label(source_kind)

        if doc_type == "statute":
            sections = self._split_statute_sections(text)
        else:
            sections = [text.strip()] if text.strip() else []

        out: List[Tuple[str, Dict[str, str]]] = []
        for section in sections:
            heading = self._section_heading(section)
            if len(section) <= chunk_size:
                body = section
                meta = {
                    "source": source,
                    "source_display": source_display,
                    "source_kind": source_kind,
                    "doc_type": doc_type,
                    "section_heading": heading,
                    "subchunk_index": "",
                }
                doc_text = (
                    f"[Источник: {source_display} | {kind_label}]\n"
                    f"[Фрагмент: {heading}]\n\n{body}"
                )
                if len(doc_text) >= self.min_chunk_len:
                    out.append((doc_text, meta))
                continue

            subchunks = self._chunk_semantic(section, chunk_size, overlap)
            for i, sub in enumerate(subchunks):
                body = sub.strip()
                if len(body) < self.min_chunk_len:
                    continue
                enriched = f"{heading}\n\n{body}"
                meta = {
                    "source": source,
                    "source_display": source_display,
                    "source_kind": source_kind,
                    "doc_type": doc_type,
                    "section_heading": heading,
                    "subchunk_index": str(i),
                }
                doc_text = (
                    f"[Источник: {source_display} | {kind_label}]\n"
                    f"[Фрагмент: {heading}]\n\n{enriched}"
                )
                out.append((doc_text, meta))
        return out

    def _resolve_path(self, path: Path, base_dir: Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    def load_corpus(
        self,
        corpus_entries: List[Dict[str, Any]],
        base_dir: Optional[Path] = None,
    ) -> None:
        """
        Загрузка нескольких файлов с метаданными. Пропускает загрузку, если коллекция не пуста.
        corpus_entries: path (Path | str), source, source_display, source_kind, doc_type
        """
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        base = base_dir or Path(__file__).resolve().parent
        all_rows: List[Tuple[str, Dict[str, str]]] = []

        for entry in corpus_entries:
            raw_path = entry["path"]
            path = self._resolve_path(Path(raw_path), base)
            if not path.exists():
                raise FileNotFoundError(f"Файл корпуса не найден: {path}")

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            rows = self._build_chunks_for_file(
                text,
                source=str(entry["source"]),
                source_display=str(entry["source_display"]),
                source_kind=str(entry.get("source_kind", "unknown")),
                doc_type=str(entry.get("doc_type", "overview")),
            )
            print(f"  {path.name}: {len(rows)} чанков")
            all_rows.extend(rows)

        if not all_rows:
            raise ValueError("Корпус пуст после нарезки")

        print(f"Всего чанков: {len(all_rows)}. Создание эмбеддингов…")
        documents = [r[0] for r in all_rows]
        metadatas = [r[1] for r in all_rows]
        embeddings = self._create_embeddings_batched(documents)

        ids = [f"doc_{i}" for i in range(len(documents))]
        batch = 100
        for start in range(0, len(documents), batch):
            end = min(start + batch, len(documents))
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  В Chroma записано {end}/{len(documents)}")

        print(f"Загружено {len(documents)} фрагментов в '{self.collection_name}'")

    def load_documents(self, file_path: str, base_dir: Optional[Path] = None) -> None:
        """Обратная совместимость: один файл как корпус из одного источника."""
        base = base_dir or Path(__file__).resolve().parent
        p = self._resolve_path(Path(file_path), base)
        self.load_corpus(
            [
                {
                    "path": p,
                    "source": "single_file",
                    "source_display": p.name,
                    "source_kind": "unknown",
                    "doc_type": "overview",
                }
            ],
            base_dir=base,
        )

    @staticmethod
    def _embedding_connection_hint(exc: BaseException) -> str:
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        tail = f" Детали: {cause}" if cause else ""
        return (
            "Не удалось связаться с API OpenAI (Connection error / таймаут). "
            "Проверьте интернет, VPN (если API недоступен из вашей сети), файрвол и корпоративный прокси. "
            "Для прокси задайте HTTPS_PROXY в системе или в PowerShell: "
            "$env:HTTPS_PROXY='http://127.0.0.1:ПОРТ'. "
            "Можно увеличить OPENAI_TIMEOUT (сек) и уменьшить RAG_EMBED_BATCH_SIZE. "
            "При использовании зеркала/шлюза укажите OPENAI_BASE_URL."
            + tail
        )

    def _create_embeddings_batched(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if batch_size is None:
            batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))
        batch_size = max(1, batch_size)
        embed_retries = max(1, int(os.getenv("OPENAI_EMBED_RETRIES", "5")))
        all_emb: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            last_err: Optional[BaseException] = None
            for attempt in range(embed_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        input=batch,
                        model=self.embedding_model,
                    )
                    by_index = sorted(response.data, key=lambda d: d.index)
                    all_emb.extend([d.embedding for d in by_index])
                    break
                except (APIConnectionError, APITimeoutError) as e:
                    last_err = e
                    wait = min(30, 2**attempt)
                    print(
                        f"  Сеть/API: попытка {attempt + 1}/{embed_retries} не удалась, "
                        f"пауза {wait} с… ({e.__class__.__name__})"
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(self._embedding_connection_hint(last_err)) from last_err
        return all_emb

    def _create_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self._create_embedding(query)
        n = min(top_k, max(1, self.collection.count()))
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        documents: List[Dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                meta = {}
                if results.get("metadatas") and results["metadatas"][0]:
                    meta = dict(results["metadatas"][0][i] or {})
                documents.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "distance": results["distances"][0][i] if results.get("distances") else None,
                        "metadata": meta,
                    }
                )
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)

    from corpus_config import default_corpus_entries

    vs = VectorStore(collection_name="test_collection")
    if vs.collection.count() == 0:
        vs.load_corpus(default_corpus_entries())

    r = vs.search("Когда вступает в силу независимая гарантия?", top_k=4)
    for i, doc in enumerate(r, 1):
        print(f"\n{i}. {doc['metadata'].get('source_display', '')} | {doc['text'][:180]}…")
