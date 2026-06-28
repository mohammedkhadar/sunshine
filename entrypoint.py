import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from sunshine.cli import main

PORT = int(os.environ.get("PORT", "8080"))


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *a, **k):
        pass


def start_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()


if __name__ == "__main__":
    t = Thread(target=start_health_server, daemon=True)
    t.start()
    raise SystemExit(main(["monitor"]))
