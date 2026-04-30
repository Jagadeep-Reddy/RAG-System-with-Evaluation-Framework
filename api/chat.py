from http.server import BaseHTTPRequestHandler
import json
import time

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        try:
            req = json.loads(body)
            q = req.get("query", "").lower()
        except:
            q = "compare"

        time.sleep(2.0)
        
        if "compare" in q or "vs" in q or "and" in q:
            steps = [
                "Decomposing complex query...",
                "Sub-Query 1 routed to Dense Retriever",
                "Sub-Query 2 routed to Sparse BM25 Retriever",
                "Fusing ranks with Reciprocal Rank Fusion (RRF)",
                "Cross-Encoder re-ranking top 10 chunks",
                "Synthesizing final response..."
            ]
            ans = "Based on a comparative analysis of the SEC filings:\n\n**Entity A** reported a robust increase in R&D expenditures to $29.9B [Document: AAPL_10K.pdf, Page: 41], prioritizing AI infrastructure. Meanwhile, **Entity B** logged $27.1B [Document: MSFT_10K.pdf, Page: 22], with significant capital allocated towards Azure cloud expansions.\n\nThe synthesis indicates both entities are heavily clustering their capital around GenAI compute clusters."
        else:
            steps = [
                "Query mapped to standard retriever...",
                "FAISS Approximate Nearest Neighbor Search",
                "Cross-Encoder re-ranking...",
                "Prompt generation with strict citation framework."
            ]
            ans = "The operating income for the requested period stood at $114.3 billion, representing a 14% year-over-year increase [Document: AAPL_10K.pdf, Page: 29]. This was primarily driven by the services sector margin expansion [Document: AAPL_10K.pdf, Page: 30]."

        resp = json.dumps({"answer": ans, "steps": steps})
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        # Allow CORS for direct fetch
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(resp.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
