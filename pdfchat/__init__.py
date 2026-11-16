"""pdfchat — chat com IA fundamentado no conteúdo dos seus PDFs (RAG).

Pipeline: PDF -> extração por página -> chunking -> embeddings -> índice
vetorial + BM25 -> recuperação híbrida (RRF) -> resposta com citações.
"""

from .config import Settings
from .models import Answer, Passage, Retrieved
from .rag import ChatEngine, IngestReport

__all__ = [
    "Answer",
    "ChatEngine",
    "IngestReport",
    "Passage",
    "Retrieved",
    "Settings",
]

__version__ = "1.0.0"
