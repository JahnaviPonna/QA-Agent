# Autonomous QA Agent — Test Case + Selenium Script Generator

**Assignment:** Development of an Autonomous QA Agent for Test Case and Script Generation. (See uploaded assignment PDF). :contentReference[oaicite:1]{index=1}

## Overview
This project ingests support documents and the target `checkout.html`, builds a grounded knowledge base (vector DB), produces documentation-grounded test cases, and generates runnable Selenium (Python) scripts from selected test cases. Backend uses **FastAPI**, UI uses **Streamlit**. Vector store uses **FAISS** and embeddings from **sentence-transformers** (or optionally OpenAI).

## Repo contents
- `checkout.html` — target web page (single-page E-Shop Checkout).
- `support_docs/` — `product_specs.md`, `ui_ux_guide.txt`, `api_endpoints.json`.
- `backend/` — main FastAPI app and modules for ingestion, vectorstore, and RAG agent.
- `streamlit_app/app.py` — Streamlit UI to upload docs/html, build KB, generate test cases, and generate scripts.
- `examples/` — sample outputs.

## Requirements
Tested with Python 3.10+. Minimal dependencies:

