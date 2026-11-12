"""Step 3: document ingestion and chunking.

For the chosen corpus, reads:
  - PDF / TXT / MD files in data/<corpus>/raw/
  - single web pages listed in data/<corpus>/urls.txt
  - crawled pages in data/<corpus>/crawled.jsonl (produced by crawl.py)
splits the text into overlapping chunks and saves them to
data/<corpus>/processed/chunks.jsonl (one JSON object per line).

Run with:  python src/ingest.py --corpus fantacalcio
"""
import json
import re
from pathlib import Path

import requests
from pypdf import PdfReader

from config import SETTINGS, corpus_paths, get_corpus
from crawl import USER_AGENT, html_to_text

CHUNK_SIZE = SETTINGS["chunking"]["size"]         # words per chunk
CHUNK_OVERLAP = SETTINGS["chunking"]["overlap"]   # words shared between consecutive chunks


# ---------------------------------------------------------------------------
# 1. Loading: every loader returns a list of (page_number, text)
# ---------------------------------------------------------------------------
def load_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def load_text(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8"))]   # text files have no pages


LOADERS = {".pdf": load_pdf, ".txt": load_text, ".md": load_text}


def load_url(url: str) -> list[tuple[int, str]]:
    """Download a single web page and keep only its readable text."""
    resp = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()                      # stop on 404 / 500 errors
    title, text = html_to_text(resp.text)
    return [(1, f"{title}. {text}")]             # a web page = one "page"


def read_url_list(urls_file: Path) -> list[str]:
    """URLs listed one per line (lines starting with # are ignored)."""
    if not urls_file.exists():
        return []
    lines = urls_file.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def collect_sources(paths: dict) -> list[tuple[str, str, callable, str]]:
    """Every source is (name shown in citations, id prefix, function that loads it, header).

    The header is prepended to EVERY chunk of that source, so even the 5th chunk of
    an article still says which article it comes from and how recent it is."""
    sources = []
    if paths["raw"].exists():
        sources += [(p.name, p.stem, lambda p=p: LOADERS[p.suffix.lower()](p), "")
                    for p in sorted(paths["raw"].iterdir()) if p.suffix.lower() in LOADERS]
    sources += [(url, f"url{i}", lambda url=url: load_url(url), "")
                for i, url in enumerate(read_url_list(paths["urls"]))]
    if paths["crawled"].exists():
        for i, line in enumerate(paths["crawled"].open(encoding="utf-8")):
            page = json.loads(line)
            header = f"[{page['title']} | {page['site']} | scaricato il {page['crawled_at']}]"
            sources.append((page["url"], f"web{i}", lambda t=page["text"]: [(1, t)], header))
    return sources


# ---------------------------------------------------------------------------
# 2. Cleaning: extracted text has messy whitespace and hyphenated words
# ---------------------------------------------------------------------------
def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # "infor-\nmation" -> "information"
    text = re.sub(r"\s+", " ", text)         # collapse newlines / multiple spaces
    text = re.sub(r"\[\s*\d+(\s*[,–-]\s*\d+)*\s*\]", "", text)   # reference markers "[ 1 ]", "[3, 4]"
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Chunking: sliding window over words, with overlap
# ---------------------------------------------------------------------------
def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        chunks.append(" ".join(chunk))
        if start + size >= len(words):   # last window reached the end
            break
    return chunks


def main() -> None:
    corpus = get_corpus()
    paths = corpus_paths(corpus)
    sources = collect_sources(paths)
    if not sources:
        print(f"No documents found for corpus '{corpus}'.")
        return

    out_file = paths["chunks"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_file.open("w", encoding="utf-8") as out:
        for name, prefix, load, header in sources:
            try:
                pages = load()
            except Exception as e:   # one broken URL/file should not stop everything
                print(f"SKIPPED {name}: {e}")
                continue
            n_file = 0
            for page_num, raw_text in pages:
                text = clean(raw_text)
                if not text:
                    continue   # empty page (e.g. scanned image without text)
                # Chunk page by page so every chunk keeps an exact page number
                for i, chunk in enumerate(chunk_words(text)):
                    record = {
                        "id": f"{prefix}-p{page_num}-c{i}",
                        "source": name,
                        "page": page_num,
                        "text": f"{header} {chunk}".strip(),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_file += 1
            total += n_file
        print(f"{len(sources)} sources -> {total} chunks")

    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
