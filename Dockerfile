# Use a lightweight official Python image
FROM python:3.12-slim

# Install system dependencies needed for compiling faiss/torch (if required)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirement files first to utilize Docker layer caching
COPY requirements-dev.txt .

# Install all core python requirements + Google GenAI SDK & BeautifulSoup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir langchain-google-genai beautifulsoup4 requests langchain-experimental langchain-community

# Copy project files and folders
COPY data/ ./data/
COPY public/ ./public/
COPY src/ ./src/

# Expose the default Hugging Face Spaces port (7860)
EXPOSE 7860

# Run FastAPI app using uvicorn on the default port
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
