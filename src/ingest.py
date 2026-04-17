import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def load_sec_filings(data_dir: str) -> List[Document]:
    """Loads all PDF files from directory, maintaining page/source metadata."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            # Inject source document explicitly
            for doc in docs:
                doc.metadata['source_doc'] = filename
            documents.extend(docs)
    return documents

def fixed_size_chunking(docs: List[Document]) -> List[Document]:
    """Fixed-size character chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def semantic_chunking(docs: List[Document]) -> List[Document]:
    """Chunks documents based on cosine similarity logic between sentences."""
    embedder = OpenAIEmbeddings()
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
    # Test execution
    ingest_pipeline()
