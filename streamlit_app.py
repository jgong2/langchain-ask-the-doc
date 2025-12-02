import os
import io
from typing import List

import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import pdfplumber
import docx2txt

# --------------------------
# Config / constants
# --------------------------

BASELINE_ISO_DIR = "baseline_iso9001"

PROVIDER = st.secrets.get("PROVIDER", "openai")
EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = st.secrets.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")


# --------------------------
# Helpers: file loading
# --------------------------

def load_pdf(file) -> str:
    """Extract text from a PDF file-like object."""
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text.append(page_text)
    return "\n\n".join(text)


def load_docx(file) -> str:
    """Extract text from a DOCX file."""
    # docx2txt needs a path or file-like object; we use buffer.
    # Streamlit's UploadedFile has .read(), so we copy to BytesIO.
    data = file.read()
    buf = io.BytesIO(data)
    text = docx2txt.process(buf)
    # reset pointer so file can be reused if needed
    file.seek(0)
    return text or ""


def load_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def load_file(uploaded_file) -> str:
    """Route based on file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return load_docx(uploaded_file)
    elif name.endswith(".txt"):
        return load_txt(uploaded_file)
    else:
        return ""


# --------------------------
# Helpers: chunking & embeddings
# --------------------------

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, source_name: str) -> List[Document]:
    """Split raw text into smaller chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, ch in enumerate(chunks):
        if ch.strip():
            docs.append(
                Document(
                    page_content=ch,
                    metadata={"source": source_name, "chunk_id": i},
                )
            )
    return docs


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_baseline_iso(embeddings):
    """Load the pre-built ISO 9001 FAISS index, if it exists."""
    if not os.path.exists(BASELINE_ISO_DIR):
        return None
    try:
        db = FAISS.load_local(
            BASELINE_ISO_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return db
    except Exception as e:
        st.warning(f"Could not load ISO baseline index: {e}")
        return None


def build_vs_from_docs(docs: List[Document], embeddings) -> FAISS:
    return FAISS.from_documents(docs, embeddings)


# --------------------------
# Helpers: RAG answering
# --------------------------

def citations_from(docs: List[Document]) -> str:
    """Build a simple citation string from documents' metadata."""
    if not docs:
        return ""
    refs = []
    for d in docs:
        src = d.metadata.get("source") or "ISO9001"
        chunk_id = d.metadata.get("chunk_id")
        label = src
        if chunk_id is not None:
            label = f"{src}#chunk{chunk_id}"
        if label not in refs:
            refs.append(label)
    return ", ".join(refs)


def answer_with_rag(vector_store: FAISS, question: str) -> str:
    """Retrieve relevant chunks and answer with LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)

    context_parts = []
    for d in docs:
        src = d.metadata.get("source") or "ISO9001"
        context_parts.append(f"[{src}] {d.page_content}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a supplier quality / ISO 9001 audit assistant. "
        "Use ONLY the provided context (ISO 9001 baseline and any uploaded docs) "
        "to answer questions. If the answer is not clearly supported, say so."
    )

    user_prompt = f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer in a clear, concise way."

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    text = (resp.content or "").strip()
    refs = citations_from(docs)
    if refs:
        text += f"\n\n**Sources:** {refs}"
    return text


# --------------------------
# UI / App logic
# --------------------------

def main():
    st.set_page_config(page_title="Supplier Audit Assistant – Phase 3", page_icon="✅")
    st.title("Supplier Audit Assistant – Phase 3")
    st.markdown(
        "Baseline: **ISO 9001** is pre-ingested (FAISS). "
        "You can also upload supplier docs to combine with ISO."
    )

    embeddings = get_embeddings()
    baseline_db = load_baseline_iso(embeddings)

    if baseline_db is None:
        st.warning("⚠️ ISO baseline index not found. Make sure baseline_iso9001/ exists.")

    uploaded_files = st.file_uploader(
        "Upload one or more files (optional, to add on top of ISO)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if "vs" not in st.session_state:
        st.session_state.vs = None

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("🔨 Build / Refresh knowledge base"):
            all_docs: List[Document] = []
            # 1) From uploaded files
            if uploaded_files:
                with st.spinner("Extracting and chunking uploaded files..."):
                    for f in uploaded_files:
                        txt = load_file(f)
                        if txt and txt.strip():
                            all_docs.extend(chunk_text(txt, f.name))

            user_db = None
            if all_docs:
                with st.spinner("Embedding uploaded documents..."):
                    user_db = build_vs_from_docs(all_docs, embeddings)
                    st.success(f"Uploaded docs indexed: {len(all_docs)} chunks.")
            else:
                if uploaded_files:
                    st.warning("No text extracted from uploaded files.")

            # 2) Combine baseline + user
            db = None
            if baseline_db is not None:
                db = baseline_db

            if user_db is not None:
                if db is None:
                    db = user_db
                else:
                    db.merge_from(user_db)

            if db is None:
                st.error("No knowledge base available (no ISO index and no uploaded docs).")
            else:
                st.session_state.vs = db
                st.success("Knowledge base is ready (ISO baseline + optional uploads).")

    with c2:
        q = st.text_input(
            "Ask an audit question…",
            placeholder="e.g., What does ISO 9001 require for management review?",
        )
        if q:
            if st.session_state.vs is None:
                st.info("Click 'Build / Refresh knowledge base' first.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        answer = answer_with_rag(st.session_state.vs, q)
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"LLM call failed: {e}")


if __name__ == "__main__":
    main()

