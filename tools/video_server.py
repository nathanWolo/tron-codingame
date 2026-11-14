#!/usr/bin/env python3
"""Serve the replay page, and save what it exports.

    python tools/video_server.py            # then open http://localhost:8765/replay.html

The page (video/replay.html) plays a recorded game from video/games/ and can
encode it as an MP4 in the browser. A browser page cannot write to disk on its
own, so it sends the finished file here and this writes it to video/out/.
Listens on this machine only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, "video")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=VIDEO, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/games.json":
            names = sorted(f for f in os.listdir(os.path.join(VIDEO, "games")) if f.endswith(".json"))
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        path, _, query = self.path.partition("?")
        name = path[len("/save/"):] if path.startswith("/save/") else ""
        if not SAFE.match(name):
            self.send_error(400, "bad name")
            return
        os.makedirs(os.path.join(VIDEO, "out"), exist_ok=True)
        left = int(self.headers.get("Content-Length", "0"))
        with open(os.path.join(VIDEO, "out", name), "ab" if "append=1" in query else "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        body = json.dumps({"saved": f"video/out/{name}", "bytes": os.path.getsize(os.path.join(VIDEO, "out", name))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if self.command == "POST":
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    os.makedirs(os.path.join(VIDEO, "games"), exist_ok=True)
    print(f"http://localhost:{args.port}/replay.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
