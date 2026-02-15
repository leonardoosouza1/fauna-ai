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
from typing import Callable, Any

logger = logging.getLogger("fauna.server")

# Diretório padrão - busca build do React em locais conhecidos
POSSIBLE_PATHS = [
    Path("/frontend_dist"),                                            # Docker volume mount
    Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist",  # Local dev relative path
    Path(__file__).parent / "static",                                  # Legacy fallback
]

STATIC_DIR = None
for p in POSSIBLE_PATHS:
    if p.exists():
        STATIC_DIR = p
        logger.info(f"Serving frontend from: {STATIC_DIR}")
        break

if not STATIC_DIR:
    logger.warning("Frontend build directory not found! Dashboard will be API-only.")
    STATIC_DIR = Path("/tmp/empty_static")
    STATIC_DIR.mkdir(exist_ok=True)

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Handler HTTP que serve o dashboard React e API da simulação."""

    def __init__(self, *args, data_provider=None, **kwargs):
        self.data_provider = data_provider
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api()
        else:
            # SPA fallback: se arquivo não existe, serve index.html
            filepath = STATIC_DIR / self.path.lstrip("/")
            if not filepath.exists() and not self.path.startswith("/assets"):
                 self.path = "/index.html"
            super().do_GET()

    def _handle_api(self):
        """Roteamento simples de API."""
        try:
            if self.path == "/api/state":
                data = self.data_provider["get_state"]()
            elif self.path.startswith("/api/agent/"):
                agent_id = int(self.path.split("/")[-1])
                data = self.data_provider["get_agent"](agent_id)
            else:
                self.send_error(404, "Endpoint not found")
                return

            response = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            logger.error(f"API Error: {e}")
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        pass


def create_handler(data_provider: dict):
    """Factory para injetar data_provider no handler."""
    class Handler(DashboardHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, data_provider=data_provider, **kwargs)
    return Handler


def start_dashboard_server(
    port: int = 8765,
    get_state_callback: Callable[[], dict] = None,
    get_agent_callback: Callable[[int], dict] = None,
) -> threading.Thread:
    """
    Inicia servidor com callbacks para acesso à memória da simulação.
    """
    data_provider = {
        "get_state": get_state_callback,
        "get_agent": get_agent_callback,
    }

    handler = create_handler(data_provider)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="fauna-dashboard",
    )
    thread.start()
    logger.info(f"🌐 Dashboard API: http://localhost:{port}")

    return thread
