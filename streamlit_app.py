import os
import tempfile
from typing import List

import streamlit as st

# Loaders / splitters / vector store
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Local embeddings (FastEmbed)
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Local LLM via Ollama
from langchain_community.chat_models.ollama import ChatOllama

# Prompting & runnable utils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Supplier Audit Assistant", page_icon="🧭", layout="wide")
st.title("🧭 Supplier Audit Assistant — Ask the Doc (RAG)")
st.caption("Upload supplier PDFs/DOCX, ask audit questions, and get answers with (filename#chunkN) citations. (Fully local: embeddings + LLM)")

# -------------------------
# Constants (tunable)
# -------------------------
CHUNK_SIZE_DEFAULT = 1100
CHUNK_OVERLAP_DEFAULT = 150

# Local embedding model (FastEmbed)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Local chat model served by Ollama (make sure `ollama pull phi3:mini`)
LLM_MODEL = "phi3:mini"

K_RETRIEVAL = 4

SYSTEM_PROMPT = (
    "You are a senior Supplier Quality Auditor AI. Your job is to answer audit questions "
    "using ONLY the provided document context. If the answer is not in the context, say you "
    "don't have enough information and suggest what evidence would satisfy the audit.\n\n"
    "Rules:\n"
    "- Be concise, factual, and use audit language (e.g., objective evidence, procedure, record, control, risk).\n"
    "- Cite sources inline immediately after the sentence they support using this exact format: (filename#chunkN).\n"
    "  Example: The supplier's PPAP is signed (ppap_manual.pdf#chunk3).\n"
    "- If multiple chunks support a statement, include all: (fileA.pdf#chunk1; fileB.docx#chunk4).\n"
    "- Do not fabricate citations or content.\n"
)

# -------------------------
# Utilities
# -------------------------
def save_upload_to_temp(uploaded_file) -> str:
    """Persist an uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1]  # keep .pdf or .docx
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(uploaded_file.getbuffer())
    return path


def load_documents(files: List) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        path = save_upload_to_temp(f)
        filename = f.name
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            st.warning(f"Unsupported file type for {filename}; only PDF/DOCX are allowed.")
            continue
        file_docs = loader.load()
        # Stamp filename into metadata now
        for d in file_docs:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = filename
        docs.extend(file_docs)
    return docs


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    # add chunk indices per file for (filename#chunkN)
    counters = {}
    for d in chunks:
        fname = d.metadata.get("filename", "document")
        counters.setdefault(fname, 0)
        counters[fname] += 1
        d.metadata["chunk_id"] = counters[fname]
    return chunks


def build_vectorstore(chunks: List[Document]) -> FAISS:
    # Local, free embeddings via FastEmbed (downloads model once, then cached)
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def format_docs_for_context(docs: List[Document]) -> str:
    # Include source tag under each chunk so the LLM can cite precisely
    formatted = []
    for d in docs:
        fname = d.metadata.get("filename", "document")
        cid = d.metadata.get("chunk_id", 0)
        formatted.append(f"{d.page_content}\n\nSource: ({fname}#chunk{cid})")
    return "\n\n---\n\n".join(formatted)


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.subheader("Settings")
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=2000, value=CHUNK_SIZE_DEFAULT, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=400, value=CHUNK_OVERLAP_DEFAULT, step=10)
    top_k = st.number_input("Top-k retrieval", min_value=1, max_value=10, value=K_RETRIEVAL, step=1)
    st.markdown("---")
    st.markdown("**Models**")
    st.text(f"LLM (local via Ollama): {LLM_MODEL}")
    st.text(f"Embeddings (local): {EMBED_MODEL}")
    st.markdown("---")
    st.info("This app runs fully local for indexing and answers. Make sure the Ollama server is running and the model is pulled.")

# -------------------------
# File upload & indexing
# -------------------------
uploaded_files = st.file_uploader(
    "Upload supplier documents (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

build_clicked = st.button("Build / Rebuild Index", type="primary", disabled=not uploaded_files)

if build_clicked:
    with st.spinner("Loading, chunking, and embedding documents…"):
        docs = load_documents(uploaded_files)
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        vs = build_vectorstore(chunks)
        st.session_state["vectorstore"] = vs
        st.session_state["all_chunks"] = chunks
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# -------------------------
# Q&A Section
# -------------------------
question = st.text_input("Ask an audit question (e.g., 'Is there a documented control plan for incoming inspection?')")
ask_clicked = st.button("Ask")

if ask_clicked:
    if "vectorstore" not in st.session_state:
        st.warning("Please upload files and click 'Build / Rebuild Index' first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": int(top_k)})

        def _format_docs(dlist: List[Document]) -> str:
            return format_docs_for_context(dlist)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question: {question}\n\nContext (chunks with sources):\n{context}\n\n"
                "Write a clear answer for a supplier audit. Use the inline citation format (filename#chunkN) right after the statements they support."
            ),
        ])

        # Local LLM (Ollama)
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

        chain = {
            "question": RunnablePassthrough(),
            "context": retriever | _format_docs,
        } | prompt | llm | StrOutputParser()

        try:
            with st.spinner("Thinking…"):
                answer = chain.invoke(question)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.info("Retrieval succeeded. Here are the top sources so you can still see relevant content:")

        # Show sources actually retrieved (even if the LLM call fails)
        with st.expander("Sources used in retrieval"):
            retrieved_docs: List[Document] = retriever.get_relevant_documents(question)
            seen = set()
            for d in retrieved_docs:
                fname = d.metadata.get("filename", "document")
                cid = d.metadata.get("chunk_id", 0)
                tag = f"({fname}#chunk{cid})"
                if tag in seen:
                    continue
                seen.add(tag)
                st.markdown(f"- **{tag}**\n\n> {d.page_content[:400]}…")

st.markdown("---")
st.caption("Built with Streamlit + FastEmbed + LangChain + FAISS + Ollama (local). © Supplier Audit Assistant")
