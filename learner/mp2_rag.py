"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from time import perf_counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters
HYBRID_CANDIDATE_K = int(os.environ.get("HYBRID_CANDIDATE_K", "20"))
RERANKER_MODEL    = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "0").lower() in {
    "1", "true", "yes"
}
USE_RERANK        = (
    os.environ.get("USE_RERANK", os.environ.get("USE_RERANKER", "0"))
    .lower()
    in {
    "1", "true", "yes"
    }
)

_sparse_chunks: list[dict[str, Any]] | None = None
_bm25 = None
_reranker = None

openai = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPEN_AI_BASE_URL"),
)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    TODO:
      - Iterate over every *.txt file in corpus_dir (use Path.glob)
      - For each file, read its text and extract the first non-empty line as title
      - Return the list of doc dicts
    """
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
        documents.append({"source": path.name, "title": title, "text": text})
    return documents


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    TODO:
      - Pick an approach
      - Implement it
      - Return list of chunk dicts
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", doc["text"].strip()) if part.strip()]
    chunks = []
    section = doc["title"]
    buffer: list[str] = []

    def add_chunk(text: str, chunk_section: str) -> None:
        chunks.append({
            "source": doc["source"],
            "title": doc["title"],
            "section": chunk_section,
            "text": text,
        })

    def flush() -> None:
        if buffer:
            add_chunk("\n\n".join(buffer), section)
            buffer.clear()

    for paragraph in paragraphs:
        if paragraph == doc["title"]:
            continue
        if "\n" not in paragraph and len(paragraph) <= 100 and not re.search(r"[.!?:]$", paragraph):
            flush()
            section = paragraph
            continue

        if len(paragraph) <= TARGET_CHUNK_SIZE:
            candidate = "\n\n".join((*buffer, paragraph))
            if buffer and len(candidate) > TARGET_CHUNK_SIZE:
                flush()
            buffer.append(paragraph)
            continue

        flush()
        for start in range(0, len(paragraph), TARGET_CHUNK_SIZE - CHUNK_OVERLAP):
            add_chunk(paragraph[start:start + TARGET_CHUNK_SIZE], section)

    flush()
    return chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    TODO:
      - Call openai.embeddings.create with EMBEDDING_MODEL and the texts
      - Extract the embedding vectors from the response
    """
    if not texts:
        return []
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection.

    TODO:
      - Use qdrant.recreate_collection
      - VectorParams with EMBEDDING_DIM and Distance.COSINE
    """
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    TODO:
      - Call embed_texts on the chunk texts
      - Build PointStruct objects (id=uuid, vector, payload=chunk dict)
      - qdrant.upsert
    """
    if not chunks:
        return
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=chunk,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def _get_bm25() -> tuple[Any, list[dict[str, Any]]]:
    """Build the local BM25 index lazily from the same chunks as Qdrant."""
    global _bm25, _sparse_chunks
    if _bm25 is None or _sparse_chunks is None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid search requires rank-bm25. Install it with "
                "pip install rank-bm25."
            ) from exc

        documents = load_corpus(CORPUS_DIR)
        _sparse_chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document)
        ]
        tokenized = [chunk["text"].lower().split() for chunk in _sparse_chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _sparse_chunks


def _hybrid_retrieve(query: str, k: int) -> list[dict[str, Any]]:
    """Fuse dense and BM25 rankings using reciprocal rank fusion."""
    candidate_k = max(k, HYBRID_CANDIDATE_K)
    dense_results = _dense_retrieve(query, candidate_k)
    bm25, sparse_chunks = _get_bm25()
    sparse_scores = bm25.get_scores(query.lower().split())
    sparse_order = sorted(
        range(len(sparse_chunks)),
        key=lambda index: sparse_scores[index],
        reverse=True,
    )[:candidate_k]

    fused: dict[str, dict[str, Any]] = {}
    rank_constant = 60

    for rank, chunk in enumerate(dense_results, start=1):
        key = f"{chunk['source']}::{chunk['text']}"
        fused[key] = {**chunk, "dense_score": chunk["score"], "hybrid_score": 1 / (rank_constant + rank)}

    for rank, index in enumerate(sparse_order, start=1):
        chunk = sparse_chunks[index]
        key = f"{chunk['source']}::{chunk['text']}"
        if key not in fused:
            fused[key] = {**chunk, "score": 0.0, "dense_score": 0.0, "hybrid_score": 0.0}
        fused[key]["sparse_score"] = float(sparse_scores[index])
        fused[key]["hybrid_score"] += 1 / (rank_constant + rank)

    return sorted(
        fused.values(), key=lambda chunk: chunk["hybrid_score"], reverse=True
    )[:candidate_k]


def _get_reranker() -> Any:
    """Load the cross-encoder only when reranking is explicitly enabled."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install it with pip install sentence-transformers."
            ) from exc
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    TODO:
      - Embed the query
      - qdrant.search with the query vector, limit=k
      - Return list of chunk dicts (include score for citations)
    """
    if USE_HYBRID_SEARCH:
        results = _hybrid_retrieve(query, k)
    else:
        candidate_k = max(k, HYBRID_CANDIDATE_K) if USE_RERANK else k
        results = _dense_retrieve(query, candidate_k)

    if USE_RERANK and results:
        reranker = _get_reranker()
        pairs = [[query, chunk["text"]] for chunk in results]
        rerank_scores = reranker.predict(pairs)
        for chunk, score in zip(results, rerank_scores):
            chunk["rerank_score"] = float(score)
        results = sorted(
            results, key=lambda chunk: chunk["rerank_score"], reverse=True
        )[:k]
    return results[:k]


def _dense_retrieve(query: str, k: int) -> list[dict[str, Any]]:
    query_vector = embed_texts([query])[0]
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
        )
        points = response.points
    else:
        points = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=k,
        )
    return [{**(point.payload or {}), "score": point.score} for point in points]


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. Use ONLY the provided excerpts; do not rely
on outside knowledge. First identify every part of the question, then answer each
part explicitly. For identity questions, include the person's name and alias,
occupation or reputation, every listed crime, and any physical identifying mark
or trademark found in the excerpts. Preserve important names and facts in the
source's wording where possible; reproduce comma-separated lists verbatim. If a
requested fact is not in the excerpts, say so plainly instead of guessing. Cite
the source (story title + section) in your answer."""


def answer(question: str, k: int = 5) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    TODO:
      - Call retrieve(question, k=k)
      - Format the retrieved chunks into a context string
        (include "[Source: <title> — <section>]" before each)
      - Call openai.chat.completions.create with SYSTEM_PROMPT and the user message
      - Return dict with: question, answer, citations, latency_ms
    """
    started = perf_counter()
    retrieved = retrieve(question, k=k)
    context = "\n\n".join(
        f"[Source: {chunk['title']} — {chunk['section']}]\n{chunk['text']}"
        for chunk in retrieved
    )
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
    )
    citations = [
        {
            "source": chunk["source"],
            "title": chunk["title"],
            "section": chunk["section"],
            "score": chunk["score"],
        }
        for chunk in retrieved
    ]
    return {
        "question": question,
        "answer": response.choices[0].message.content or "",
        "citations": citations,
        "latency_ms": round((perf_counter() - started) * 1000),
    }


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=5)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
