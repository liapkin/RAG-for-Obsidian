import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import INDEX_NPZ
from .indexing import build_index
from .llm import stream_deepseek
from .query import build_query

PAGE = (Path(__file__).parent / "index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages, sources = build_query(data["question"], data.get("history"))
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Connection", "close")  # no content-length: stream until close
        self.end_headers()
        for delta in stream_deepseek(messages):
            self.wfile.write(delta.encode())
            self.wfile.flush()
        src = "\n".join(f"- {s}" for s in sources)
        self.wfile.write(f"\n\n<details><summary>Sources</summary>\n\n{src}\n\n</details>".encode())

    def log_message(self, *a):
        pass


def serve(port=8000):
    if not INDEX_NPZ.exists():
        build_index()
    host = os.environ.get("HOST", "127.0.0.1")  # compose sets 0.0.0.0
    print(f"serving on http://localhost:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
