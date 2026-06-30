# Production RAG Troubleshooting & Implementation Q&A

This document details the engineering challenges, error logs, diagnostic analyses, and exact code changes implemented during the development and deployment of the Agentic SEC RAG Explorer.

---

## 📋 Table of Contents
1. [In-Memory Ingestion of Raw SEC HTML Filings](#q1-in-memory-ingestion-of-raw-sec-html-filings)
2. [Gemini API 429 Errors During Document Ingestion](#q2-gemini-api-429-errors-during-document-ingestion)
3. [Temporal Prompt Context Calibration](#q3-temporal-prompt-context-calibration)
4. [GitHub Mermaid Diagram Render Failures](#q4-github-mermaid-diagram-render-failures)
5. [GitHub Language Stats Showing 99.7% HTML](#q5-github-language-stats-showing-997-html)
6. [Hugging Face Spaces Deployment Crashing with ModuleNotFoundError](#q6-hugging-face-spaces-deployment-crashing-with-modulenotfounderror)
7. [Vercel Deployment Returning Mock Answers](#q7-vercel-deployment-returning-mock-answers)
8. [Multi-Hop Queries Failing with 429 Rate Limits](#q8-multi-hop-queries-failing-with-429-rate-limits)

---

### Q1: How was the data ingestion pipeline upgraded to load real HTML SEC EDGAR filings instead of low-accuracy mock PDF files?

**Answer:**
SEC reports are natively published in `.htm`/`.html` format. We replaced `PyPDFLoader` with `BSHTMLLoader` configured with the native Python `html.parser` to clean out raw markup tags. Additionally, chunk size was increased to `15,000` characters with a `1,500` character overlap using `RecursiveCharacterTextSplitter` to keep complex financial tables and disclosures intact inside single document nodes.

#### Code Changes in `src/ingest.py`:
```diff
-from langchain_community.document_loaders import PyPDFLoader
+from langchain_community.document_loaders import BSHTMLLoader
 from langchain_text_splitters import RecursiveCharacterTextSplitter
 from langchain_experimental.text_splitter import SemanticChunker

 def load_sec_filings(data_dir: str) -> List[Document]:
     # ...
-    # Old PDF loader
-    loader = PyPDFLoader(filepath)
-    docs.extend(loader.load())
+    # Upgraded HTML loader with built-in python html parser
+    loader = BSHTMLLoader(filepath, bs_kwargs={"features": "html.parser"})
+    loaded_docs = loader.load()
+    for doc in loaded_docs:
+        doc.metadata["source_doc"] = filename
+        doc.metadata["page"] = 1  # HTML filings are unified single-page docs
+    docs.extend(loaded_docs)
 
 def fixed_size_chunking(docs: List[Document]) -> List[Document]:
-    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
+    # Large window size to avoid tearing tables
+    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1500)
     return text_splitter.split_documents(docs)
```

---

### Q2: How did you bypass the 15 Requests Per Minute (RPM) free-tier rate limit on the Gemini Embedding API during batch ingestion?

**Answer:**
Standard LangChain embedding wrappers like `GoogleGenerativeAIEmbeddings` execute document embedding queries concurrently inside a thread pool, which counts each text chunk as an individual API request. This immediately triggered `429 Resource Exhausted` exceptions on our 53 document chunks. 

We implemented `RESTBatchEmbeddings` inside `src/retrieval.py` which directly interfaces with the native Google Generative Language REST endpoint `batchEmbedContents`. This packages up to 100 document chunks inside a single JSON request payload, meaning the entire ingestion database is built in exactly **one API call**.

#### Code Changes in `src/retrieval.py`:
```python
import requests
from langchain_core.embeddings import Embeddings

class RESTBatchEmbeddings(Embeddings):
    """Embeddings implementation using native Google REST API for batching."""
    def __init__(self, model_name: str = "models/gemini-embedding-2"):
        self.model_name = model_name
        self.api_key = os.environ.get("GOOGLE_API_KEY")

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
                    {"model": self.model_name, "content": {"parts": [{"text": text}]}}
                    for text in chunk
                ]
            }
            res = requests.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            embeddings = [e["values"] for e in data["embeddings"]]
            all_embeddings.extend(embeddings)
            
        return all_embeddings
```

---

### Q3: Why was the LLM generating incorrect dates or failing to contextualize timeline queries like "Compare 2023 R&D spending", and how was it solved?

**Answer:**
The raw chunks inside the SEC filings contained relative timelines and table columns without explicit year references in every single text block. To calibrate the model's temporal baseline, we updated the system prompt inside `src/generation.py` instructing the model to assume that all facts and figures in `AAPL_10K.html` and `MSFT_10K.html` belong to **fiscal year 2023** unless specified otherwise.

#### Code Changes in `src/generation.py`:
```diff
         self.qa_prompt = ChatPromptTemplate.from_messages([
             ("system", """You are a meticulous financial analyst. 
             You must answer the user's question using ONLY the provided context blocks below.
             
             CRITICAL INSTRUCTIONS:
             1. Every single factual claim you make MUST end with a bracketed citation pointing to its source.
             2. The citation format must be exactly: [Document: <source_doc>, Page: <page>].
             3. If the context does not contain enough information to answer the question, state: "Insufficient context to answer."
             4. Do not invent metrics or facts.
+            5. Note that the documents provided (e.g., AAPL_10K.html, MSFT_10K.html) are the annual reports (10-K filings) for the fiscal year 2023. Therefore, unless another year is explicitly stated in the text, you can assume the reported figures in these documents are for the fiscal year 2023.
             
             Context Blocks:
             {context}"""),
```

---

### Q4: Why did the architecture diagram fail to render on the GitHub README.md, throwing a "Parse error: got 'AMP'" message?

**Answer:**
GitHub uses a strict Mermaid rendering parser. Subgraph header blocks containing spaces and raw characters like `&` (e.g., `subgraph Hybrid Retrieval & Reranking`) broke the parser because the `&` character is reserved. 

We updated the diagram to define subgraphs with unique, clean alphanumeric IDs (e.g., `subgraph_retrieval` and `subgraph_generation`) and enclosed their display labels in double quotes `["Title & Name"]`.

#### Code Changes in `README.md`:
```diff
-    subgraph Hybrid Retrieval & Reranking
+    subgraph subgraph_retrieval ["Hybrid Retrieval & Reranking"]
         H --> J[Parallel Hybrid Retriever]
         I --> J
         J -->|Semantic Match| E
         J -->|Keyword Match| F
         E --> K(Reciprocal Rank Fusion - RRF)
         F --> K
         K --> L[Cross-Encoder Reranker]
     end

-    style Hybrid Retrieval & Reranking fill:#0d1117...
+    style subgraph_retrieval fill:#0d1117,stroke:#21262d,stroke-width:2px,color:#c9d1d9
```

---

### Q5: Why did GitHub classify the repository as "99.7% HTML" despite all RAG logic being written in Python, and how was this corrected?

**Answer:**
GitHub Linguist determines repository language statistics by comparing the raw file size in bytes of all repository files. Because our raw SEC HTML filing files in `data/` were large (totaling **11.4 Megabytes**), they completely dominated our Python source code files (totaling only **20 Kilobytes**).

We created a `.gitattributes` file at the root of the repository to mark all data HTML files as non-detectable by Linguist, instantly restoring **Python** as the primary language.

#### Code Changes in `.gitattributes` (New File):
```text
# Exclude raw financial data filings from GitHub language statistics
data/*.html linguist-detectable=false
data/**/*.html linguist-detectable=false
```

---

### Q6: Why did the Hugging Face Spaces Docker container fail to startup, throwing `ModuleNotFoundError: No module named 'langchain_experimental'`?

**Answer:**
The RAG pipeline imports experimental libraries (`SemanticChunker`) and community loaders (`BSHTMLLoader`) in `src/ingest.py`. While these were installed in the local virtual environment, they were missing from the production `Dockerfile` pip command, causing the Hugging Face python container to crash on initialization.

We explicitly added `langchain-experimental` and `langchain-community` to the `pip install` list inside the `Dockerfile`.

#### Code Changes in `Dockerfile`:
```diff
 # Install all core python requirements + Google GenAI SDK & BeautifulSoup
 RUN pip install --no-cache-dir --upgrade pip && \
     pip install --no-cache-dir -r requirements-dev.txt && \
-    pip install --no-cache-dir langchain-google-genai beautifulsoup4 requests
+    pip install --no-cache-dir langchain-google-genai beautifulsoup4 requests langchain-experimental langchain-community
```

---

### Q7: Why did the live Vercel deployment link display mock responses even after the backend was fully operational?

**Answer:**
Vercel is a serverless platform. To bypass Vercel's strict 250MB deployment size limit (which prevents installing massive packages like PyTorch and sentence-transformers), Vercel's backend functions are mapped to a mocked handler in `api/chat.py`. 

To allow users to access the real RAG backend from the Vercel link, we updated `public/script.js` to dynamically identify the browser hostname. When accessed locally, it queries localhost. When hosted in production on Vercel, it routes requests directly to the live Hugging Face Space backend (which is fully capable of running PyTorch, sentence-transformers, and Qdrant).

#### Code Changes in `public/script.js`:
```diff
     try {
-        const response = await fetch('/api/chat', {
+        const apiBase = location.hostname === 'localhost' || location.hostname === '127.0.0.1'
+            ? ''
+            : 'https://jagadeep24-rag-system-with-evaluation-framework.hf.space';
+        
+        const response = await fetch(`${apiBase}/api/chat`, {
             method: 'POST',
             headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify({ query: text })
         });
```

---

### Q8: Why did complex multi-hop queries fail and fallback to the mock template, throwing a `429 Client Error: Too Many Requests` in the server logs?

**Answer:**
When a complex query was decomposed (e.g. comparing Apple and Microsoft), `src/agent_router.py` used `ThreadPoolExecutor` to execute sub-queries in parallel threads concurrently. This caused multiple `embed_query` requests to hit the Google Embedding API at the exact same millisecond. Since the free-tier Gemini key enforces strict concurrency and rate limit thresholds, it returned a 429 error.

We resolved this by:
1. Converting parallel thread execution in `src/agent_router.py` to **sequential loops** with a 1-second delay.
2. Building an **exponential backoff retry loop** inside the custom `RESTBatchEmbeddings` class.

#### Code Changes in `src/agent_router.py`:
```diff
-        # Execute sub-queries in parallel
-        with ThreadPoolExecutor() as executor:
-            sub_answers = list(executor.map(self._process_sub_query, sub_queries))
+        # Execute sub-queries sequentially to avoid concurrent rate limit errors on the free-tier API
+        import time
+        sub_answers = []
+        for sq in sub_queries:
+            sub_answers.append(self._process_sub_query(sq))
+            time.sleep(1.0) # Grace period to prevent triggering rapid RPM rate limits
```

#### Code Changes in `src/retrieval.py`:
```python
    def embed_query(self, text: str) -> List[float]:
        import time
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model_name,
            "content": {"parts": [{"text": text}]}
        }
        
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                res = requests.post(url, json=payload)
                if res.status_code == 429:
                    print(f"Embedding query got 429. Retrying in {backoff}s...")
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
```
