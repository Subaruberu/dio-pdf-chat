"""Atalho: `python ingest.py inputs [--rebuild]`."""

from __future__ import annotations

import sys

from pdfchat.cli import main

if __name__ == "__main__":
    args = sys.argv[1:] or ["inputs"]
    if "--no-chat" not in args and "--chat" not in args:
        args.append("--no-chat")
    raise SystemExit(main(args))
