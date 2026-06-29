import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def load_sec_filings(data_dir: str) -> List[Document]:
    """Loads all PDF and HTML files from directory, maintaining page/source metadata."""
    documents = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
            # Inject source document explicitly
            for doc in docs:
                doc.metadata['source_doc'] = filename
            documents.extend(docs)
        elif filename.endswith(".htm") or filename.endswith(".html"):
            loader = BSHTMLLoader(path, bs_kwargs={"features": "html.parser"})
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_doc'] = filename
                # Default page metadata to 1 for HTML documents
                doc.metadata['page'] = 1
            documents.extend(docs)
    return documents

def fixed_size_chunking(docs: List[Document]) -> List[Document]:
    """Fixed-size character chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1500)
    return splitter.split_documents(docs)

def semantic_chunking(docs: List[Document]) -> List[Document]:
    """Chunks documents based on cosine similarity logic between sentences."""
    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    # Drops chunk when sequence meaning changes via embedding similarity
    splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")
    return splitter.split_documents(docs)

def ingest_pipeline(data_dir: str = "data/") -> List[Document]:
    """Orchestrates loading and chunking."""
    print("Loading PDFs...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Please place SEC 10-K PDFs in {data_dir} directory.")
        return []
        
    raw_docs = load_sec_filings(data_dir)
    print(f"Loaded {len(raw_docs)} pages.")
    
    print("Chunking documents (Fixed size)...")
    chunks = fixed_size_chunking(raw_docs)
    print(f"Created {len(chunks)} chunks.")
    
    return chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents to persistent local Qdrant database")
    parser.add_argument("--data-dir", default="data/", help="Directory containing PDFs/HTMLs")
    parser.add_argument("--qdrant-path", default="data/qdrant_db", help="Path to local Qdrant database")
    parser.add_argument("--bm25-path", default="data/bm25_index.pkl", help="Path to BM25 pickle file")
    args = parser.parse_args()
    
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import pickle
    
    chunks = ingest_pipeline(args.data_dir)
    if chunks:
        print("Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("Generating embeddings for all chunks...")
        texts = [doc.page_content for doc in chunks]
        vectors = embeddings.embed_documents(texts)
        
        print(f"Indexing {len(chunks)} chunks into Qdrant at '{args.qdrant_path}'...")
        os.makedirs(os.path.dirname(args.qdrant_path), exist_ok=True)
        client = QdrantClient(path=args.qdrant_path)
        
        # Recreate collection
        client.recreate_collection(
            collection_name="sec_filings",
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        
        # Prepare points
        points = []
        for idx, (doc, vector) in enumerate(zip(chunks, vectors)):
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
            
        client.upsert(collection_name="sec_filings", points=points)
        print("Qdrant index built successfully!")
        
        print("Building BM25 Sparse Index...")
        from langchain_community.retrievers import BM25Retriever
        sparse_retriever = BM25Retriever.from_documents(chunks)
        
        print(f"Saving BM25 index to '{args.bm25_path}'...")
        os.makedirs(os.path.dirname(args.bm25_path), exist_ok=True)
        with open(args.bm25_path, "wb") as f:
            pickle.dump(sparse_retriever, f)
        print("BM25 index saved successfully!")

