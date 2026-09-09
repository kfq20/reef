"""A local shim that strips the ``openai/`` provider prefix litellm requires.

Terminus 2 reaches the model through litellm, which demands a provider-prefixed
model name (``openai/macaron-v1-tall``) even for a plain OpenAI-compatible
endpoint, while the endpoint itself wants the bare name. Reef's own model
binding renders the same name into the episode config as it uses for its own
calls, so both names must work on one URL.

This shim listens on a local port, forwards to the upstream, and rewrites the
request body's ``model`` field: ``openai/X`` becomes ``X``. Nothing else is
touched. Run it as:

    python -m recipes.mint_terminus.harness.openai_shim 8971 https://mintcn.macaron.xin sk-...
"""

from __future__ import annotations

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM: str = ""
API_KEY: str = ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; the reef logs carry the story
        pass

    def _forward(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("model"), str) and parsed["model"].startswith("openai/"):
            parsed["model"] = parsed["model"].removeprefix("openai/")
            body = json.dumps(parsed).encode("utf-8")
        request = urllib.request.Request(
            f"{UPSTREAM}{self.path}",
            data=body or None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}},
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in {"content-length", "transfer-encoding", "connection"}:
                        continue
                    self.send_header(key, value)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("content-type", exc.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, OSError) as exc:
            payload = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = _forward


def main() -> int:
    global UPSTREAM, API_KEY
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8971
    UPSTREAM = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "https://mintcn.macaron.xin"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"openai shim on 127.0.0.1:{port} -> {UPSTREAM}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
