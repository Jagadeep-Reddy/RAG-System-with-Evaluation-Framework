import os
from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
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
            res = requests.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            embeddings = [e["values"] for e in data["embeddings"]]
            all_embeddings.extend(embeddings)
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model_name,
            "content": {
                "parts": [{"text": text}]
            }
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["embedding"]["values"]

class HybridRetriever:
    """
    Production Hybrid Retriever combining Dense (FAISS) and Sparse (BM25)
    with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
    """
    
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.embeddings = RESTBatchEmbeddings(model_name="models/gemini-embedding-2")
        
        # 1. Initialize Dense Retriever (FAISS)
        print("Building Dense FAISS Index...")
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # 2. Initialize Sparse Retriever (BM25)
        print("Building Sparse BM25 Index...")
        self.sparse_retriever = BM25Retriever.from_documents(self.docs)
        self.sparse_retriever.k = 10
        
        # 3. Initialize Cross-Encoder
        # Best tradeoff between performance and latency
        print("Loading Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
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
        dense_res = self.dense_retriever.invoke(query)
        sparse_res = self.sparse_retriever.invoke(query)
        
        # Filter down via hybrid
        rrf_res = self._rrf(dense_res, sparse_res)
        
        # Rerank top ~20 with cross encoder down to top_n
        final_res = self._cross_encoder_rerank(query, rrf_res, top_n=top_n)
        return final_res
