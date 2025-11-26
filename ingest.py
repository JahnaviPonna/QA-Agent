# backend/ingest.py
import os
from bs4 import BeautifulSoup
import json
from typing import List, Dict
from pathlib import Path

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_html_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")
        # Keep visible text and attributes useful for selectors
        text = soup.get_text(separator="\n")
        # Also include id/name/class attributes in a structured manner
        attrs = []
        for el in soup.find_all():
            info = {"tag": el.name}
            if el.attrs:
                for k, v in el.attrs.items():
                    info[k] = v
                attrs.append(info)
        extra = "\n\nHTML_ELEMENTS_METADATA:\n" + json.dumps(attrs, ensure_ascii=False)
        return text + "\n\n" + extra

def parse_json_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_and_store_documents(paths: List[str]):
    """
    For each file produce dict: {'text':..., 'metadata':{'source': filename}}
    """
    docs = []
    for p in paths:
        p = Path(p)
        name = p.name
        if p.suffix.lower() in [".md", ".txt"]:
            txt = read_text_file(str(p))
        elif p.suffix.lower() in [".html", ".htm"]:
            txt = parse_html_file(str(p))
        elif p.suffix.lower() in [".json"]:
            txt = parse_json_file(str(p))
        elif p.suffix.lower() in [".pdf"]:
            # For brevity, not implementing pdf parsing here. In practice use pymupdf or unstructured.
            txt = f"[PDF parsing placeholder] {name}"
        else:
            txt = read_text_file(str(p))
        docs.append({"text": txt, "metadata": {"source_document": name}})
    return docs
