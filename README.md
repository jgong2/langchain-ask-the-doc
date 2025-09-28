# Supplier Audit Assistant (Local-Only)

Reads supplier PDFs/DOCX and answers audit questions with inline citations `(filename#chunkN)`.

**How it runs**
- Embeddings: FastEmbed (`BAAI/bge-small-en-v1.5`) – local
- LLM: Ollama (`phi3:mini`) – local
- No OpenAI key needed (we hit 429 quota earlier and went fully local)

## Use it (on my machine)
This demo runs on my laptop. If you want to try it, message me and I’ll open a time window.

## Run it yourself (macOS)
```bash
# 1) go to folder
cd ~/Documents/GitHub/langchain-ask-the-doc
# 2) activate venv
source .venv/bin/activate
# 3) start the app
streamlit run streamlit_app.py
# stop with Ctrl+C
