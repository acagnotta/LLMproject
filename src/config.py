"""Loads the YAML configuration in config/.

  config/settings.yaml          global settings (LLM, chunking, crawler, paths)
  config/corpora/<name>.yaml    one file per knowledge base ("corpus")

To add a new corpus, just add a new YAML file in config/corpora/.
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent   # project folder
CONFIG_DIR = ROOT / "config"

SETTINGS = yaml.safe_load((CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8"))
CORPORA = {
    f.stem: yaml.safe_load(f.read_text(encoding="utf-8"))
    for f in sorted((CONFIG_DIR / "corpora").glob("*.yaml"))
}

DATA_DIR = ROOT / SETTINGS["paths"]["data_dir"]
DB_DIR = str(ROOT / SETTINGS["paths"]["db_dir"])


def corpus_paths(corpus: str) -> dict[str, Path]:
    base = DATA_DIR / corpus
    return {
        "raw": base / "raw",                    # PDF / TXT / MD files
        "urls": base / "urls.txt",              # single URLs, one per line
        "crawled": base / "crawled.jsonl",      # output of crawl.py
        "chunks": base / "processed" / "chunks.jsonl",
    }


def get_corpus() -> str:
    """Read --corpus from the command line (default from settings.yaml)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA, default=SETTINGS["default_corpus"])
    return parser.parse_args().corpus
