# LLM Document Assistant (RAG)

A question-answering assistant over a collection of documents, built with Retrieval-Augmented Generation and a local open-source LLM.

> 🚧 Work in progress

## Tech stack
- **LLM:** Llama 3.2 (3B) running locally via [Ollama](https://ollama.com)
- **Embeddings:** sentence-transformers
- **Vector store:** ChromaDB
- **UI:** Streamlit

## Setup
```bash
conda env create -f environment.yml
conda activate llm

# Install Ollama from https://ollama.com/download, then:
ollama pull llama3.2:3b
python src/hello_llm.py
```

## Project structure
```
src/         source code
data/raw/    input documents (not versioned)
notebooks/   experiments
tests/       tests and evaluation
```
