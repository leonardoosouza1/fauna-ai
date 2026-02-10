"""
Servidor HTTP para o Dashboard — Projeto Fauna.

Serve o dashboard web e fornece os dados da simulação via API REST.
Usa apenas a stdlib do Python (sem Flask/FastAPI).
"""

import http.server
import json
import os
import threading
import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger("fauna.server")

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = "simulation_data"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Handler HTTP que serve o dashboard e dados da simulação."""

    def __init__(self, *args, data_dir=DATA_DIR, **kwargs):
        self.data_dir = data_dir
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_state()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif self.path == "/style.css":
            self._serve_file("style.css", "text/css")
        elif self.path == "/app.js":
            self._serve_file("app.js", "application/javascript")
        else:
            super().do_GET()

    def _serve_state(self):
        """Serve o estado atual da simulação como JSON."""
        state_path = os.path.join(self.data_dir, "state.json")
        try:
            with open(state_path, "r") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "waiting"}')

    def _serve_file(self, filename, content_type):
        """Serve um arquivo estático."""
        filepath = STATIC_DIR / filename
        try:
            with open(filepath, "r") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(content.encode())
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Silenciar logs do HTTP server."""
        pass


def create_handler(data_dir: str):
    """Factory para criar handler com data_dir customizado."""
    class Handler(DashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_dir=data_dir, **kwargs)
    return Handler


def start_dashboard_server(
    port: int = 8765,
    data_dir: str = DATA_DIR,
) -> threading.Thread:
    """
    Inicia o servidor do dashboard em uma thread separada.

    Args:
        port: Porta HTTP
        data_dir: Diretório com os dados da simulação

    Returns:
        Thread do servidor
    """
    handler = create_handler(data_dir)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="fauna-dashboard",
    )
    thread.start()
    logger.info(f"🌐 Dashboard: http://localhost:{port}")

    return thread
