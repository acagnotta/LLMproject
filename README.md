# LLM Document Assistant (RAG)

A question-answering assistant that answers **only from a chosen set of sources** (PDFs, web pages or entire websites), built with Retrieval-Augmented Generation and a local open-source LLM. Everything runs on-device: no API keys, no costs.

Two knowledge bases ("corpora") are included:
- **`fantacalcio`** — Italian fantasy-football assistant, built by crawling fantacalcio.it and sosfanta.com (injuries, line-ups, team news).
- **`papers`** — foundational LLM papers (Transformer, BERT, RAG) plus a couple of Wikipedia pages.

> 🚧 Work in progress — built step by step as a learning project.

## How it works
```
         OFFLINE (run once / when sources change)
websites ──crawl.py──► crawled.jsonl ─┐
PDF / URLs ───────────────────────────┴─ingest.py─► chunks ──index.py─► ChromaDB (vectors)

         ONLINE (every question)
question ─┬─► semantic search (embeddings) ─┐
           └─► keyword search (BM25) ────────┴─► fusion (RRF) ─► 20 candidates ─► reranker ─► top 5
                                                                                            │
                         answer + sources ◄── Qwen 2.5 7B ◄── prompt [system + sources + question]
```
1. **Crawling** — reads the websites' sitemaps, respects `robots.txt`, waits between requests, keeps only useful sections.
2. **Ingestion** — extracts and cleans text, splits it into overlapping chunks (300 words, 50 overlap).
3. **Indexing** — turns each chunk into a 384-dim embedding (sentence-transformers) and stores it in ChromaDB.
4. **Hybrid retrieval** — semantic search (good with paraphrases) + BM25 keyword search (good with player names), merged with Reciprocal Rank Fusion, then re-ordered by a multilingual cross-encoder reranker.
5. **Generation** — the LLM answers *only* from the top chunks, citing `[Source n]`; real URLs/pages are printed by the code, not by the LLM. Each chunk carries a header (title, site, download date) so the model can prefer the most recent info.

## Roadmap
- [x] **Step 1** — Environment setup (conda, git)
- [x] **Step 2** — First calls to a local LLM, interactive chat, temperature experiments
- [x] **Step 3** — Document ingestion and chunking (PDF, single URLs, sitemap-based crawler)
- [x] **Step 4** — Embeddings + vector search and answer generation with sources
- [ ] **Step 5** — Web UI with Streamlit
- [ ] **Step 6** — Evaluation (retrieval recall@k, answer quality)

## Tech stack
- **LLM:** Qwen 2.5 (7B, quantized) running locally via [Ollama](https://ollama.com) — Llama 3.2 3B was tried first but hallucinated more
- **Embeddings:** sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2` for Italian, `all-MiniLM-L6-v2` for English)
- **Vector store:** ChromaDB
- **Keyword search / reranking:** rank_bm25, cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`
- **Crawling / parsing:** requests, BeautifulSoup, pypdf
- **Configuration:** YAML
- **UI:** Streamlit *(Step 5)*

## Setup
```bash
conda env create -f environment.yml
conda activate llm

# Install Ollama from https://ollama.com/download, then:
ollama pull qwen2.5:7b
```

## Usage
Build a knowledge base and chat with it (`--corpus` defaults to `fantacalcio`):
```bash
python src/crawl.py  --corpus fantacalcio   # download pages from the configured websites (~10 min)
python src/ingest.py --corpus fantacalcio   # clean + chunk
python src/index.py  --corpus fantacalcio   # embed + store in ChromaDB
python src/rag.py    --corpus fantacalcio   # ask questions
```
For the `papers` corpus, download the PDFs into `data/papers/raw/` first ([1706.03762](https://arxiv.org/abs/1706.03762), [1810.04805](https://arxiv.org/abs/1810.04805), [2005.11401](https://arxiv.org/abs/2005.11401)) and skip `crawl.py`.

The fantacalcio data changes every week: re-run the first three commands to refresh it.

### Learning scripts (Step 2)
| Script | What it does |
|---|---|
| `python src/hello_llm.py` | Single question/answer; shows how the system prompt changes the output and how many tokens are used |
| `python src/chat.py` | Interactive terminal chat with conversation memory and streaming output (`/temp`, `/history`, `/reset`, `/bye`) |
| `python src/temperature_demo.py` | Asks the same question at different temperatures to show the effect of sampling |

## Configuration
Everything configurable lives in `config/`, no code changes needed:

| File | What you can change |
|---|---|
| `config/settings.yaml` | LLM model, `top_k`, chunk size/overlap, crawler delay and user agent |
| `config/corpora/<name>.yaml` | websites to crawl (sitemaps, URL filter, max pages), embedding model, system prompt |

To create a new assistant on a different topic, add a new file in `config/corpora/` — it becomes available as `--corpus <name>`.

## Example
```
You: Quali giocatori della Juventus sono infortunati e quando rientrano?
Assistant: Secondo [Source 1], gli infortunati della Juventus sono: Andrea Cambiaso ...
           Weston McKennie ... Zeki Celik ... Inoltre, secondo [Source 2], Kalulu ...
Sources:
  [1] https://www.fantacalcio.it/serie-a/squadre/juventus  (distance 0.282)
  [2] https://www.sosfanta.com/squadra/juventus/juve-nuovo-stop-per-cambiaso-...  (distance 0.282)

You: Chi ha vinto il Pallone d'Oro 2010?
Assistant: Non ho trovato questa informazione nelle fonti.   ← refuses to use outside knowledge
```

## Improvements log
Six test questions on the fantacalcio corpus (injuries, new coach, suspensions, standings, line-up advice):

| Version | Correct & grounded | Main issue |
|---|---|---|
| Llama 3.2 3B + semantic search | 3 / 6 | hallucinations, player names confused (Pulisic ≈ Pavlovic), outdated matchdays |
| + Qwen 2.5 7B, hybrid search (BM25), only latest matchdays, chunk headers | 4 / 6 | relevant table chunks ranked 4th–6th, just outside top-5 |
| + cross-encoder reranking | 5 / 6 | indirect mentions ("Lauti") not recognized |

## Known limitations
- **Tables lose their structure** when converted to plain text (e.g. the penalty-takers page becomes a list of names without teams), so table-based questions are answered poorly.
- **Local 7B model:** answers are sometimes vague, and small details can be invented (e.g. a wrong surname).
- **Partial coverage:** only selected sections and the last 7 days of news are crawled, not the whole websites (700k+ pages).
- **Static pages only:** pages rendered with JavaScript would need a headless browser.
- Crawled content belongs to the respective websites; it is stored locally for personal use and is not redistributed in this repository.

## Project structure
```
config/
  settings.yaml          global settings
  corpora/               one YAML per knowledge base
src/
  config.py              loads the YAML configuration
  crawl.py               sitemap-based polite crawler
  ingest.py              loading, cleaning, chunking
  index.py               embeddings + ChromaDB
  rag.py                 retrieval + generation chat
  hello_llm.py, chat.py, temperature_demo.py   Step 2 learning scripts
data/<corpus>/           raw files, crawled pages, chunks (not versioned)
chroma_db/               vector database (not versioned)
environment.yml          conda environment
requirements.txt         direct dependencies
requirements-lock.txt    pinned versions (pip freeze)
```
