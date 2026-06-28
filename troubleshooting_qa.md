# RAG System Troubleshooting Q&A Document

This document records the major runtime errors, deployment bugs, and configuration issues encountered during the development and deployment of the Agentic SEC RAG Explorer, along with their matching resolutions.

---

### Q1: Why did the ingestion pipeline trigger `429 Resource Exhausted` rate-limit errors during text chunk embedding?
* **Problem**: LangChain's default `GoogleGenerativeAIEmbeddings` wrapper uses concurrent thread pools to embed lists of text chunks. On the free tier, this concurrent blast of single HTTP requests exceeds the 15 Requests Per Minute (RPM) limit instantly, throwing 429 errors.
* **Fix**: Created a custom `RESTBatchEmbeddings` class inside [src/retrieval.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/retrieval.py) that bypasses LangChain's concurrent thread pool. It communicates directly with Google's native REST endpoint `batchEmbedContents`, bundling up to 100 document chunks in a **single network call**, completely avoiding rate limits.

---

### Q2: Why did `BSHTMLLoader` fail to load the HTML filings on startup?
* **Problem**: By default, BeautifulSoup loaders inside LangChain try to use `lxml` as the default parser. `lxml` depends on compiled C header files which were not present on the host system, causing the module to fail.
* **Fix**: Configured the document loader inside [src/ingest.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/ingest.py) to use Python's built-in parser by passing `bs_kwargs={"features": "html.parser"}` to the initialization call.

---

### Q3: Why did GitHub display the error `Unable to render rich display` in the README's Mermaid diagram?
* **Problem**: The subgraph names in the original Mermaid flowchart contained spaces and special characters (e.g., `subgraph Hybrid Retrieval & Reranking`). The Mermaid engine on GitHub rejects raw special characters in subgraph headers and styling calls, causing rendering crashes.
* **Fix**: Rewrote the diagram in [README.md](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/README.md) to use strict alphanumeric subgraph IDs (e.g., `subgraph_retrieval`) and wrapped the display headers in quotes `["Hybrid Retrieval & Reranking"]`.

---

### Q4: Why did GitHub classify the repository as 99.7% HTML instead of Python?
* **Problem**: The project directory contains two official SEC 10-K HTML files (`data/MSFT_10K.html` and `data/AAPL_10K.html`) that total 11.4 MB in size. Since our Python source files are only ~20 KB, GitHub's automatic Linguist engine flagged the repository as HTML based on file sizes.
* **Fix**: Created a [`.gitattributes`](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/.gitattributes) file in the root of the workspace instructing GitHub's compiler to ignore all `.html` data filings from statistics:
  ```text
  data/*.html linguist-detectable=false
  data/**/*.html linguist-detectable=false
  ```

---

### Q5: Why did the Hugging Face Space fail to start up with the error `ModuleNotFoundError: No module named 'langchain_experimental'`?
* **Problem**: The text splitter in [src/ingest.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/ingest.py) imports `SemanticChunker` which is kept under LangChain's experimental packages. This module was not installed in the Docker container's environment.
* **Fix**: Modified the [`Dockerfile`](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/Dockerfile) to explicitly append `langchain-experimental` and `langchain-community` to the pip installation layer.

---

### Q6: Why did the Hugging Face Space git push fail with a `YAML metadata verification` rejection?
* **Problem**: Hugging Face Spaces validates the YAML frontmatter at the top of `README.md`. The initial parameter `colorTo: violet` was rejected because `violet` is not in Hugging Face's restricted list of colors.
* **Fix**: Changed the color parameter to `colorTo: purple` in [README.md](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/README.md) to comply with validation checks.

---

### Q7: Why did complex comparative queries fail with a 429 error and trigger mock fallbacks while simple queries succeeded?
* **Problem**: Decomposed sub-queries were being executed in parallel using `ThreadPoolExecutor`. The concurrent calls made concurrent API calls to `models/gemini-embedding-2:embedContent`. Since Gemini's free tier blocks concurrent requests, it rejected them with 429 errors.
* **Fix**:
  1. Converted parallel thread execution to sequential loop execution with a `1.0s` sleep pause in [src/agent_router.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/agent_router.py) to prevent rate limits.
  2. Implemented a robust retry wrapper using exponential backoff (starting at `2 seconds` and doubling, up to 5 attempts) in the custom `RESTBatchEmbeddings` class inside [src/retrieval.py](file:///c:/Users/jagadheep%20reddy/Desktop/Production-RAG/src/retrieval.py) so that transient 429 requests are retried instead of causing a crash.
