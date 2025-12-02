# Phase 3 - ISO 9001 Ingestion Script
# This script will load the ISO 9001 PDF, chunk it, add metadata, and store it in a baseline FAISS index.

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import pdfplumber

# Placeholder paths (we will update these later)
ISO_PDF_PATH = "ISO9001.pdf"
BASELINE_DB_DIR = "baseline_iso9001"

def load_pdf(path):
    """Load PDF text using pdfplumber."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages):
            pages.append((idx + 1, page.extract_text()))
    return pages

def basic_chunk(pages):
    """Simple chunking for now (we will upgrade to clause-aware chunking)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    docs = []
    for page_num, text in pages:
        chunks = text_splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "chunk_id": idx,
                    }
                )
            )
    return docs

def embed_and_save(docs):
    """Embed the chunks and save FAISS index."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(BASELINE_DB_DIR)
    print(f"Saved FAISS index to: {BASELINE_DB_DIR}")

def main():
    if not os.path.exists(ISO_PDF_PATH):
        print("ERROR: ISO PDF not found. Place ISO9001.pdf in the project folder.")
        return
    
    print("Loading ISO 9001 PDF...")
    pages = load_pdf(ISO_PDF_PATH)

    print(f"Loaded {len(pages)} pages.")
    print("Chunking...")
    docs = basic_chunk(pages)

    print(f"Created {len(docs)} chunks.")
    print("Embedding and saving FAISS index...")
    embed_and_save(docs)

    print("Done. ISO baseline created.")

if __name__ == "__main__":
    main()

