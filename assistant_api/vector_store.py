"""
Модуль работы с векторным хранилищем ChromaDB.
Загрузка нескольких источников с метаданными; нарезка с учётом структуры источника — см. chunking.py.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import time
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError

try:
    from .chunking import ChunkingConfig, build_chunks, kind_label
    from .corpus_config import index_settings
    from .openai_client import get_openai_client
except ImportError:
    from chunking import ChunkingConfig, build_chunks, kind_label
    from corpus_config import index_settings
    from openai_client import get_openai_client

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


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
        settings = index_settings()
        self.embedding_model = settings["embedding_model"]
        self.chunk_size = settings["chunk_size"]
        self.chunk_overlap = settings["chunk_overlap"]
        self.min_chunk_len = settings["min_chunk_len"]

    def _kind_label(self, source_kind: str) -> str:
        return kind_label(source_kind)

    def _build_chunks_for_file(
        self,
        text: str,
        source: str,
        source_display: str,
        source_kind: str,
        doc_type: str,
        chunker: str = "generic",
        citation: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Нарезка одного источника по его структуре: список (текст для индекса, метаданные)."""
        config = ChunkingConfig(self.chunk_size, self.chunk_overlap, self.min_chunk_len)
        chunks = build_chunks(
            text,
            source=source,
            source_display=source_display,
            source_kind=source_kind,
            doc_type=doc_type,
            chunker=chunker,
            citation=citation,
            config=config,
        )
        return [(chunk.text, chunk.metadata) for chunk in chunks]

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
        corpus_entries: path (Path | str), source, source_display, source_kind, doc_type, chunker, citation
        """
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        base = base_dir or Path(__file__).resolve().parent
        all_rows: List[Tuple[str, Dict[str, Any]]] = []

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
                chunker=str(entry.get("chunker", "generic")),
                citation=entry.get("citation"),
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
        try:
            for start in range(0, len(documents), batch):
                end = min(start + batch, len(documents))
                self.collection.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
                print(f"  В Chroma записано {end}/{len(documents)}")
        except BaseException:
            # Частично записанная коллекция при следующем запуске выглядела бы как «уже загруженная».
            self._reset_collection()
            raise

        print(f"Загружено {len(documents)} фрагментов в '{self.collection_name}'")

    def _reset_collection(self) -> None:
        """Пересоздать пустую коллекцию с теми же параметрами."""
        try:
            self.client.delete_collection(name=self.collection_name)
        finally:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

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
