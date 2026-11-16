"""Atalho: `python chat.py` abre o chat no terminal."""

from __future__ import annotations

import sys

from pdfchat.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
