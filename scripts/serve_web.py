"""
Local Development Server for the Speech Emotion Recognition Web UI.

Usage:
  python scripts/serve_web.py
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)


def main():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print(f"  Speech Emotion Recognition Web UI is live!")
        print(f"  Open in browser: {url}")
        print("  Press Ctrl+C to stop the server.")
        print("=" * 60)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
