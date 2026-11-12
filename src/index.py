"""Step 4a: embed the chunks and store them in a vector database (ChromaDB).

Run once (and again whenever documents change):
    python src/index.py --corpus fantacalcio
"""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import CORPORA, DB_DIR, corpus_paths, get_corpus


def main() -> None:
    corpus = get_corpus()
    chunks_file = corpus_paths(corpus)["chunks"]
    chunks = [json.loads(line) for line in chunks_file.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks for corpus '{corpus}'")

    # 1. Text -> vectors. Similar meaning => vectors pointing in similar directions.
    model = SentenceTransformer(CORPORA[corpus]["embed_model"])
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,   # unit length, so dot product == cosine similarity
        show_progress_bar=True,
    )
    print(f"Embedding shape: {embeddings.shape}")   # (n_chunks, 384)

    # 2. Store vectors + text + metadata in a persistent vector DB (a folder on disk).
    #    One collection per corpus, so knowledge bases never mix.
    client = chromadb.PersistentClient(path=DB_DIR)
    if corpus in [c.name for c in client.list_collections()]:
        client.delete_collection(corpus)   # rebuild from scratch
    collection = client.create_collection(corpus, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    print(f"Indexed {collection.count()} chunks into {DB_DIR}/ (collection '{corpus}')")


if __name__ == "__main__":
    main()
