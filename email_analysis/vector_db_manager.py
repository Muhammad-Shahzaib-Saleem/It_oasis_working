import os
import json
import uuid
import sqlite3
from typing import List, Dict, Any
import threading

import faiss                         # pip install faiss-cpu  (or faiss-gpu)
from sentence_transformers import SentenceTransformer

from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


class VectorDBManager:
    """Manage FAISS vector database for email storage and retrieval."""

    def __init__(self):
        # folders / files
        self.db_dir        = os.path.join(VECTOR_DB_PATH, COLLECTION_NAME)
        self.index_file    = os.path.join(self.db_dir, "faiss.index")
        self.meta_file     = os.path.join(self.db_dir, "meta.sqlite")
        self.id_map_file   = os.path.join(self.db_dir, "id_map.json")  # new: store list of UUIDs by FAISS index

        # embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.dim             = self.embedding_model.get_sentence_embedding_dimension()

        # runtime handles
        self.index = None           # faiss.IndexFlatIP
        self._faiss_lock = threading.Lock()  # Lock for thread-safe FAISS writes
        self.id_map = []           # list of UUID strings, index = FAISS vector index

        self._initialize_db()
        self._load_id_map()

    def _initialize_db(self):
        """Create dir, load or build FAISS index, and set up metadata store."""
        os.makedirs(self.db_dir, exist_ok=True)

        # 1) FAISS index (cosine using L2‑normalised vectors)
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # inner‑product
            faiss.write_index(self.index, self.index_file)

        # 2) Initialize SQLite DB (create tables if needed)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS emails (
                       id            TEXT PRIMARY KEY,
                       document      TEXT,
                       metadata_json TEXT
                   )"""
            )

    def _conn(self):
        """Create a new SQLite connection (one per thread/operation)."""
        return sqlite3.connect(self.meta_file, timeout=30)

    def _load_id_map(self):
        """Load FAISS index to UUID mapping from JSON file."""
        if os.path.exists(self.id_map_file):
            with open(self.id_map_file, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

    def _save_id_map(self):
        """Save FAISS index to UUID mapping to JSON file."""
        with open(self.id_map_file, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize(vecs):
        """L2‑normalise so dot‑product == cosine similarity."""
        faiss.normalize_L2(vecs)
        return vecs

    def add_emails(self, emails: List[Dict[str, Any]]) -> bool:
        """
        Insert new emails.
        Each `email` dict must contain a 'text_content' field; other keys become metadata.
        """
        try:
            documents, metadatas, ids = [], [], []
            for entry in emails:
                text = entry.get("text_content", "")
                if not text:
                    continue
                documents.append(text)
                metadatas.append({k: str(v) for k, v in entry.items() if k != "text_content"})
                ids.append(str(uuid.uuid4()))

            if not documents:
                return False

            # embeddings
            embs = self.embedding_model.encode(
                documents, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            ).astype("float32")
            embs = self._normalize(embs)

            # add to FAISS with thread-safe lock
            with self._faiss_lock:
                self.index.add(embs)
                faiss.write_index(self.index, self.index_file)

            # persist metadata with new DB connection
            with self._conn() as conn:
                conn.executemany(
                    "INSERT INTO emails (id, document, metadata_json) VALUES (?, ?, ?)",
                    [
                        (id_, doc, json.dumps(meta, ensure_ascii=False))
                        for id_, doc, meta in zip(ids, documents, metadatas)
                    ],
                )

            # update id_map and save it
            self.id_map.extend(ids)
            self._save_id_map()

            print(f"Successfully added {len(documents)} emails to vector database")
            return True

        except Exception as e:
            print(f"Error adding emails to vector database: {e}")
            return False

    def search_emails(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        try:
            if self.index.ntotal == 0:
                print("No emails in vector database")
                return []

            q_vec = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
            q_vec = self._normalize(q_vec)

            distances, indices = self.index.search(q_vec, n_results)
            distances, indices = distances[0], indices[0]

            results = []
            with self._conn() as conn:
                for idx, score in zip(indices, distances):
                    if idx == -1:
                        continue

                    if idx >= len(self.id_map):
                        print(f"Warning: FAISS index {idx} out of range of id_map length {len(self.id_map)}")
                        continue

                    email_id = self.id_map[idx]
                    row = conn.execute(
                        "SELECT id, document, metadata_json FROM emails WHERE id = ?",
                        (email_id,),
                    ).fetchone()
                    if row:
                        results.append({
                            "id": row[0],
                            "document": row[1],
                            "metadata": json.loads(row[2]) if row[2] else {},
                            "score": float(score),
                        })
                    else:
                        print(f"No email found for id {email_id}")

            return results

        except Exception as e:
            print(f"[search_emails] Error: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Return basic info about the FAISS index."""
        try:
            return {
                "name": COLLECTION_NAME,
                "count": self.index.ntotal,
                "path": self.db_dir,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}

    def clear_collection(self) -> bool:
        """Remove all vectors and metadata but keep the on‑disk structure."""
        try:
            with self._faiss_lock:
                self.index.reset()
                faiss.write_index(self.index, self.index_file)

            with self._conn() as conn:
                conn.execute("DELETE FROM emails")

            # Clear id_map and save
            self.id_map = []
            self._save_id_map()

            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def delete_database(self) -> bool:
        """Delete every index/metadata file and start fresh."""
        try:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            if os.path.exists(self.meta_file):
                os.remove(self.meta_file)
            if os.path.exists(self.id_map_file):
                os.remove(self.id_map_file)

            self._initialize_db()
            self.id_map = []
            return True
        except Exception as e:
            print(f"Error deleting database: {e}")
            return False
