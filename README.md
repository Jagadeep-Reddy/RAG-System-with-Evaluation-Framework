# Agentic SEC RAG Explorer (Production-Grade)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](https://www.python.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-8E75C2?style=flat&logo=googlegemini)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Dense_Search-00A4EF?style=flat)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-ready, low-latency Retrieval-Augmented Generation (RAG) system specialized in analyzing and comparing official SEC 10-K filings. The platform features query decomposition routing, hybrid dense/sparse indexing, cross-encoder reranking, and a native REST batch-embedding engine designed to bypass rate limits under free-tier API quotas.

A beautiful dashboard UI is served locally at [http://localhost:8000/](http://localhost:8000/).

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[Raw SEC 10-K filings in data/] --> B(BSHTMLLoader + html.parser)
        B --> C[RecursiveCharacterTextSplitter / 15k char window]
        C --> D[RESTBatchEmbeddings]
        D -->|Single batch HTTP call| E[(In-Memory FAISS Dense Index)]
        C --> F[(In-Memory BM25 Sparse Index)]
    end

    subgraph Agentic Query Routing
        Q[User Query] --> G(Agent Router / gemini-2.5-flash)
        G -->|Decompose & Parallelize| H[Sub-Query 1: Apple metrics]
        G -->|Decompose & Parallelize| I[Sub-Query 2: Microsoft metrics]
    end

    subgraph Hybrid Retrieval & Reranking
        H --> J[Parallel Hybrid Retriever]
        I --> J
        J -->|Semantic Match| E
        J -->|Keyword Match| F
        E --> K(Reciprocal Rank Fusion - RRF)
        F --> K
        K --> L[Cross-Encoder Reranker]
    end

    subgraph Generation & Consensus
        L --> M[System QA Prompt / Strict Citation Rules]
        M --> N(Self-Consistency Voter / Batch Temp 0.7)
        N --> O[Final Synthesized Answer w/ UI Citation Badges]
    end

    subgraph User Presentation
        O --> P[FastAPI Server]
        P -->|Mounts public/ directory| UI[Interactive HTML/CSS/JS Dashboard]
    end

    style Data Ingestion fill:#0d1117,stroke:#21262d,stroke-width:2px,color:#c9d1d9
    style Agentic Query Routing fill:#0d1117,stroke:#21262d,stroke-width:2px,color:#c9d1d9
    style Hybrid Retrieval & Reranking fill:#0d1117,stroke:#21262d,stroke-width:2px,color:#c9d1d9
    style Generation & Consensus fill:#0d1117,stroke:#21262d,stroke-width:2px,color:#c9d1d9
    style User Presentation fill:#161b22,stroke:#30363d,stroke-width:2px,color:#c9d1d9
```

---

## 🛠️ Core Engineering Features

### 1. High-Performance REST Batch Embeddings
LangChain's default `GoogleGenerativeAIEmbeddings` utilizes a thread pool to send individual chunks concurrently, which quickly triggers `429 Resource Exhausted` rate-limit exceptions on Gemini API keys (limit of 15 requests per minute). 
We implement `RESTBatchEmbeddings` inside [src/retrieval.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/retrieval.py) to directly communicate with the native Google REST endpoint `batchEmbedContents`. This packages up to 100 document chunks in a **single payload/network request**, reducing 53 chunk embeddings to a single network call.

### 2. Native SEC HTML Loader
Since SEC EDGAR files are natively published in HTML format, our ingestion pipeline in [src/ingest.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/ingest.py) reads `.htm` / `.html` documents directly using `BSHTMLLoader` wrapped with the built-in python `"html.parser"`. Old, low-accuracy mock PDF files have been deprecated.

### 3. Agentic Query Decomposer & Router
Complex comparison queries are split into single-topic sub-queries using `agent_router.py`. Each sub-query runs parallel semantic (FAISS) and lexical (BM25) searches, which are fused using **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
A **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`) evaluates joint token representations of the query and candidate passages, filtering out noise and bubbles the most relevant sections to the generation node.

### 4. Hallucination Detection & Strict Citation Persona
Our prompt guidelines enforce strict bracketed citations linked directly to raw HTML files (e.g. `[Document: AAPL_10K.html, Page: 22]`). We implement self-consistency checks using a higher temperature batch execution (`temperature=0.7`, 3 paths). If generating divergent facts, a warning is raised.

---

## 📂 Project Structure

```text
Production-RAG/
├── data/                    # Raw SEC 10-K HTML filings (AAPL, MSFT)
├── public/                  # Static web dashboard resources
│   ├── index.html           # Front-end structure
│   ├── style.css            # Stylesheets (custom themes, dark mode)
│   └── script.js            # Frontend chat interface & citation parser
├── src/                     # Core application logic
│   ├── api.py               # FastAPI router endpoints & mounting
│   ├── ingest.py            # Document loading & character chunking
│   ├── retrieval.py         # REST batching, FAISS, BM25, RRF, Reranker
│   ├── agent_router.py      # Query decomposition & routing logic
│   └── generation.py        # LCEL chain, prompt context, self-consistency
├── tests/                   # Verification suite
├── requirements-dev.txt     # Developer tools (pytest, black)
└── requirements-ui.txt      # Runtime dependencies
```

---

## ⚙️ Configuration & Environment

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=AIzaSy...       # Your Google AI Studio API Key
CHUNK_TYPE=fixed              # fixed or semantic
```

---

## 🚀 Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements-ui.txt
pip install -r requirements-dev.txt
```

### 2. Start the FastAPI Application
Execute uvicorn to start the local server. Ingestion, chunking, and index construction will trigger automatically on the first chat query request:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 3. Access the Dashboard
Open your browser and navigate to:
👉 **[http://localhost:8000/](http://localhost:8000/)**

---

## 🧪 Testing the API

You can test the RAG server programmatically.

### cURL Request:
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Compare Apple and Microsoft R&D spending in 2023"}'
```

### Response Schema:
```json
{
  "answer": "In 2023, Apple's Research and development (R&D) expense was $29,915 million [Document: AAPL_10K.html, Page: 22]. Microsoft's R&D spending was $27,195 million [Document: MSFT_10K.html, Page: 47]. This represents approximately $2.72 billion more spent by Apple.",
  "steps": [
    "Decomposing complex query...",
    "Sub-Query 1: 'What was Apple's R&D spending in 2023?'",
    "Sub-Query 2: 'What was Microsoft's R&D spending in 2023?'",
    "Running parallel dense/sparse hybrid retrieval...",
    "Reciprocal Rank Fusion (RRF) & Cross-Encoder reranking...",
    "Synthesizing final multi-hop response."
  ]
}
```

---

## 📈 Evaluation & Observability

### RAGAS Integration
We evaluate the quality of responses across four major metrics:
* **Faithfulness**: Verifies if the answer is derived strictly from context.
* **Answer Relevancy**: Verifies if the answer directly addresses the user query.
* **Context Precision**: Measures whether the retrieved documents match ground truth ordering.
* **Context Recall**: Verifies if the retrieval system recovered all necessary information fragments.

### LangSmith Tracing
To trace latency, LLM paths, and RRF rank details, set the following environment variables:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=sec-qa-production
LANGCHAIN_API_KEY=your-langsmith-key
```

---

## 🛡️ Production Deployment Guidelines

For deploying this application in production:
1. **WSGI/ASGI Server**: Run uvicorn behind a process manager like Gunicorn with Uvicorn workers:
   ```bash
   gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```
2. **Reverse Proxy**: Place the application behind Nginx to handle SSL termination, rate-limiting, and static file caching for the `public/` folder.
3. **Containerization**: Use a multi-stage Dockerfile containing caching for Python wheels and lightweight base images (`python:3.12-slim`).
