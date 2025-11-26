# backend/vectorstore.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle
from typing import List, Dict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, store_path="vectorstore.db"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.store_path = store_path
        self.index = None
        self.metadatas = []
        self.ids = []
        # lazy init
        if os.path.exists(store_path):
            self._load()

    def reset(self):
        self.index = None
        self.metadatas = []
        self.ids = []
        if os.path.exists(self.store_path):
            os.remove(self.store_path)

    def _save(self):
        data = {"metadatas": self.metadatas, "ids": self.ids}
        with open(self.store_path + ".meta", "wb") as f:
            pickle.dump(data, f)
        faiss.write_index(self.index, self.store_path + ".index")

    def _load(self):
        meta_path = self.store_path + ".meta"
        idx_path = self.store_path + ".index"
        if os.path.exists(meta_path) and os.path.exists(idx_path):
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.metadatas = data["metadatas"]
                self.ids = data["ids"]
            self.index = faiss.read_index(idx_path)

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        # simple char chunker
        chunks = []
        i = 0
        while i < len(text):
            end = i + chunk_size
            chunk = text[i:end]
            chunks.append(chunk)
            i = end - overlap
        return chunks

    def add_documents(self, documents: List[Dict]):
        texts = []
        metas = []
        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            for c in chunks:
                texts.append(c)
                metas.append(doc.get("metadata", {}))
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        start_id = len(self.ids)
        for i, m in enumerate(metas):
            self.ids.append(start_id + i)
            self.metadatas.append(m)
        self._save()

    def query(self, query_text: str, top_k=5):
        emb = self.model.encode([query_text], convert_to_numpy=True)
        if self.index is None:
            return []
        D, I = self.index.search(emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadatas):
                results.append(self.metadatas[idx])
        return results
