import os
from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Assuming cross-encoder is available via sentence-transformers
from sentence_transformers import CrossEncoder

import requests
from langchain_core.embeddings import Embeddings

class RESTBatchEmbeddings(Embeddings):
    """Embeddings implementation using native Google REST API for batching."""
    def __init__(self, model_name: str = "models/gemini-embedding-2"):
        self.model_name = model_name
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        if not texts:
            return []
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:batchEmbedContents?key={self.api_key}"
            payload = {
                "requests": [
                    {
                        "model": self.model_name,
                        "content": {
                            "parts": [{"text": text}]
                        }
                    }
                    for text in chunk
                ]
            }
            
            max_retries = 5
            backoff = 2.0
            chunk_embeddings = None
            
            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json=payload)
                    if res.status_code == 429:
                        print(f"Embedding batch got 429. Retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    res.raise_for_status()
                    data = res.json()
                    chunk_embeddings = [e["values"] for e in data["embeddings"]]
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(backoff)
                    backoff *= 2
            
            if chunk_embeddings:
                all_embeddings.extend(chunk_embeddings)
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        import time
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model_name,
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                res = requests.post(url, json=payload)
                if res.status_code == 429:
                    print(f"Embedding query got 429. Retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                res.raise_for_status()
                data = res.json()
                return data["embedding"]["values"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2

class HybridRetriever:
    """
    Production Hybrid Retriever combining Dense (Qdrant) and Sparse (BM25)
    with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
    """
    
    def __init__(self, docs: List[Document] = None, qdrant_path: str = "data/qdrant_db", bm25_path: str = "data/bm25_index.pkl"):
        self.docs = docs
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, VectorParams, Distance
        import pickle
        import os
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.qdrant_client = QdrantClient(path=qdrant_path)
        
        if docs is not None and len(docs) > 0:
            # 1. Initialize Dense Retriever (Qdrant) dynamically
            print("Building Dense Index in Qdrant (Dynamic)...")
            os.makedirs(os.path.dirname(qdrant_path), exist_ok=True)
            
            # Embed documents
            texts = [doc.page_content for doc in docs]
            vectors = self.embeddings.embed_documents(texts)
            
            # Recreate collection
            self.qdrant_client.recreate_collection(
                collection_name="sec_filings",
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
            )
            
            # Upsert
            points = []
            for idx, (doc, vector) in enumerate(zip(docs, vectors)):
                points.append(
                    PointStruct(
                        id=idx,
                        vector=vector,
                        payload={
                            "page_content": doc.page_content,
                            "source_doc": doc.metadata.get("source_doc", "Unknown"),
                            "page": doc.metadata.get("page", 1)
                        }
                    )
                )
            self.qdrant_client.upsert(collection_name="sec_filings", points=points)
            
            # 2. Initialize Sparse Retriever (BM25) dynamically
            print("Building Sparse BM25 Index (Dynamic)...")
            self.sparse_retriever = BM25Retriever.from_documents(self.docs)
            self.sparse_retriever.k = 10
            
            # Save the sparse retriever to disk
            print(f"Persisting BM25 index to {bm25_path}...")
            os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
            with open(bm25_path, "wb") as f:
                pickle.dump(self.sparse_retriever, f)
        else:
            # Load from persistent storage
            print(f"Checking persistent Qdrant collection from '{qdrant_path}'...")
            try:
                if os.path.exists(qdrant_path):
                    # Verify collection exists
                    collections = self.qdrant_client.get_collections().collections
                    exists = any(c.name == "sec_filings" for c in collections)
                    if not exists:
                        raise FileNotFoundError(f"Collection 'sec_filings' not found in Qdrant database at {qdrant_path}")
                else:
                    raise FileNotFoundError(f"Persistent Qdrant database not found at {qdrant_path}. Please run the ingestion script first.")
                    
                print(f"Loading persistent BM25 index from '{bm25_path}'...")
                if os.path.exists(bm25_path):
                    with open(bm25_path, "rb") as f:
                        self.sparse_retriever = pickle.load(f)
                    self.sparse_retriever.k = 10
                else:
                    raise FileNotFoundError(f"Persistent BM25 index file not found at {bm25_path}. Please run the ingestion script first.")
            except Exception as e:
                # Release file lock before raising
                if hasattr(self, 'qdrant_client') and self.qdrant_client is not None:
                    try:
                        self.qdrant_client.close()
                    except Exception:
                        pass
                raise e
        
        # 3. Initialize Cross-Encoder
        # Best tradeoff between performance and latency
        print("Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
    def _dense_retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Performs dense vector search directly in Qdrant."""
        query_vector = self.embeddings.embed_query(query)
        search_result = self.qdrant_client.query_points(
            collection_name="sec_filings",
            query=query_vector,
            limit=k
        )
        retrieved_docs = []
        for hit in search_result.points:
            retrieved_docs.append(
                Document(
                    page_content=hit.payload["page_content"],
                    metadata={
                        "source_doc": hit.payload.get("source_doc", "Unknown"),
                        "page": hit.payload.get("page", 1)
                    }
                )
            )
        return retrieved_docs

        
    def _rrf(self, dense_results: List[Document], sparse_results: List[Document], k: int = 60) -> List[Document]:
        """
        Reciprocal Rank Fusion algorithm to merge dense and sparse results.
        score = 1 / (k + rank)
        """
        import hashlib
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        for rank, doc in enumerate(dense_results):
            source = doc.metadata.get('source_doc', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            content_hash = hashlib.md5(doc.page_content.encode('utf-8', errors='ignore')).hexdigest()
            doc_id = f"{source}_p{page}_{content_hash}"
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
            
        for rank, doc in enumerate(sparse_results):
            source = doc.metadata.get('source_doc', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            content_hash = hashlib.md5(doc.page_content.encode('utf-8', errors='ignore')).hexdigest()
            doc_id = f"{source}_p{page}_{content_hash}"
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
            
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def _cross_encoder_rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        """
        Takes top RRF candidates and computes expensive cross-attention scores.
        """
        if not docs:
            return []
            
        # Prepare inputs for CrossEncoder format (Query, Document)
        model_inputs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(model_inputs)
        
        # Pair up and sort by CrossEncoder scores
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:top_n]]

    def retrieve(self, query: str, top_n: int = 5) -> List[Document]:
        """Final pipeline: Dense/Sparse in parallel -> RRF -> Reranking"""
        dense_res = self._dense_retrieve(query, k=10)
        sparse_res = self.sparse_retriever.invoke(query)
        
        # Filter down via hybrid
        rrf_res = self._rrf(dense_res, sparse_res)
        
        # Rerank top ~20 with cross encoder down to top_n
        final_res = self._cross_encoder_rerank(query, rrf_res, top_n=top_n)
        return final_res

