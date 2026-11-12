"""Step 4b: Retrieval-Augmented Generation chat with HYBRID search + RERANKING.

For every question:
  1. semantic search: embed the question, find the closest chunks in ChromaDB
  2. keyword search: BM25 over the same chunks (great for names like "Pulisic")
  3. merge the two rankings with Reciprocal Rank Fusion
  4. rerank the candidates with a cross-encoder and keep the best TOP_K
  5. put them in the prompt (numbered as sources) and let the LLM answer

Run with:  python src/rag.py --corpus fantacalcio
"""
import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import CORPORA, DB_DIR, SETTINGS

LLM_MODEL = SETTINGS["llm_model"]
TOP_K = SETTINGS["top_k"]
CANDIDATES = 20   # chunks taken from EACH search before merging
RRF_K = 60        # standard constant of Reciprocal Rank Fusion


STOPWORDS = set("il lo la i gli le un una di a da in con su per tra fra e o ma che chi è "
                "del della dei degli delle al alla ai agli alle nel nella sul sulla come".split())


def tokenize(text: str) -> list[str]:
    """Words used by BM25: lowercase, no stopwords, crude stemming.
    'Lautaro è infortunato?' -> ['lautar', 'infort']

    Stemming = cutting words to their root, so 'infortunato', 'infortunati' and
    'infortunio' all match. Here we simply keep the first 6 letters."""
    return [w[:6] for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class RAG:
    def __init__(self, corpus: str):
        settings = CORPORA[corpus]
        self.system_prompt = settings["system_prompt"]
        # MUST be the same embedding model used in index.py
        self.embedder = SentenceTransformer(settings["embed_model"])
        self.reranker = CrossEncoder(SETTINGS["rerank_model"])
        self.collection = chromadb.PersistentClient(path=DB_DIR).get_collection(corpus)

        # BM25 needs all chunk texts in memory (fine for thousands of chunks)
        data = self.collection.get(include=["documents", "metadatas"])
        self.chunks = {
            id_: {"id": id_, "text": doc, "source": meta["source"], "page": meta["page"]}
            for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self.ids])

    def semantic_search(self, question: str, n: int) -> list[str]:
        """Chunk ids ranked by meaning (cosine similarity of embeddings)."""
        q_vec = self.embedder.encode(question, normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=[q_vec], n_results=n)
        return res["ids"][0]

    def keyword_search(self, question: str, n: int) -> list[str]:
        """Chunk ids ranked by BM25: rewards rare words (names!) that appear in the chunk."""
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in best if scores[i] > 0]

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        """Two-stage retrieval:
        1. hybrid search merged with Reciprocal Rank Fusion -> CANDIDATES chunks (fast, rough)
        2. cross-encoder reranking -> best k chunks (slow, precise)"""
        fused: dict[str, float] = {}
        for ranking in (self.semantic_search(question, CANDIDATES),
                        self.keyword_search(question, CANDIDATES)):
            for rank, id_ in enumerate(ranking, 1):
                fused[id_] = fused.get(id_, 0) + 1 / (RRF_K + rank)   # score = sum of 1/(60+rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]

        # The cross-encoder reads question AND chunk together, so it judges relevance
        # much better than comparing two separately computed embeddings.
        scores = self.reranker.predict([(question, self.chunks[i]["text"]) for i in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**self.chunks[i], "score": float(s)} for i, s in ranked]

    @staticmethod
    def build_prompt(question: str, chunks: list[dict]) -> str:
        """Augment the question with the retrieved context, numbering each source."""
        context = "\n\n".join(f"[Source {i}]\n{c['text']}" for i, c in enumerate(chunks, 1))
        return f"Context:\n{context}\n\nQuestion: {question}"

    def answer(self, question: str) -> tuple[str, list[dict]]:
        chunks = self.retrieve(question)
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_prompt(question, chunks)},
            ],
            options={
                "temperature": 0,   # factual task -> deterministic
                "num_ctx": 8192,    # context window: must fit system prompt + all chunks
            },
        )
        return response["message"]["content"], chunks


def print_sources(chunks: list[dict]) -> None:
    """Sources are printed by the code, not by the LLM: they are always correct."""
    for i, c in enumerate(chunks, 1):
        page = f" p.{c['page']}" if c["source"].endswith(".pdf") else ""
        print(f"  [{i}] {c['source']}{page}")


if __name__ == "__main__":
    from config import get_corpus

    corpus = get_corpus()
    rag = RAG(corpus)
    print(f"Corpus: {corpus} | LLM: {LLM_MODEL}. Type /bye to quit.\n")
    while (q := input("You: ").strip()) != "/bye":
        if q:
            text, chunks = rag.answer(q)
            print(f"\nAssistant: {text}\nSources:")
            print_sources(chunks)
            print()
