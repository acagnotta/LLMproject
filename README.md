# LLM Document Assistant (RAG)

A question-answering assistant over a collection of documents, built with Retrieval-Augmented Generation and a local open-source LLM. Everything runs on-device: no API keys, no costs, and documents never leave the machine.

> 🚧 Work in progress — built step by step as a learning project.

## Roadmap
- [x] **Step 1** — Environment setup (conda, git)
- [x] **Step 2** — First calls to a local LLM, interactive chat, temperature experiments
- [ ] **Step 3** — Document ingestion and chunking
- [ ] **Step 4** — Embeddings + vector search (retrieval) and answer generation
- [ ] **Step 5** — Web UI with Streamlit
- [ ] **Step 6** — Evaluation (retrieval recall@k, answer quality)

## Tech stack
- **LLM:** Llama 3.2 (3B, quantized) running locally via [Ollama](https://ollama.com)
- **Embeddings:** sentence-transformers *(Step 4)*
- **Vector store:** ChromaDB *(Step 4)*
- **UI:** Streamlit *(Step 5)*

## Setup
```bash
conda env create -f environment.yml
conda activate llm

# Install Ollama from https://ollama.com/download, then:
ollama pull llama3.2:3b
```

## Usage
| Script | What it does |
|---|---|
| `python src/hello_llm.py` | Single question/answer; shows how the system prompt changes the output and how many tokens are used |
| `python src/chat.py` | Interactive terminal chat with conversation memory and streaming output |
| `python src/temperature_demo.py` | Asks the same question at different temperatures to show the effect of sampling |

**Chat commands:** `/temp <value>` change temperature · `/history` show messages/tokens sent · `/reset` clear memory · `/bye` quit

## Project structure
```
src/
  hello_llm.py          Step 2: single LLM call
  chat.py               Step 2: interactive chat with memory
  temperature_demo.py   Step 2: temperature experiment
data/raw/               input documents (not versioned)
notebooks/              experiments
tests/                  tests and evaluation
environment.yml         conda environment
requirements.txt        direct dependencies
requirements-lock.txt   pinned versions (pip freeze)
```
