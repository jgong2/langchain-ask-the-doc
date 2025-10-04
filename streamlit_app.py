# streamlit_app.py  —  Phase 2.1 (OpenAI-only)
import os
import io
import streamlit as st

# LangChain bits
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --------------------------
# Page / constants
# --------------------------
st.set_page_config(page_title="Supplier Audit Assistant — OpenAI (Phase 2.1)", page_icon="📄", layout="wide")
st.title("📄 Supplier Audit Assistant — Phase 2.1 (OpenAI only)")

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150
EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
CHAT_MODEL  = st.secrets.get("OPENAI_CHAT_MODEL",  os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o-mini"))

# --------------------------
# Ensure OpenAI key is set
# --------------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
else:
    st.error("OPENAI_API_KEY is missing. Add it to .streamlit/secrets.toml")
    st.stop()

with st.sidebar:
    st.markdown("### How to use")
    st.markdown("1) Upload PDFs/DOCX/TXT\n2) Click **Build index**\n3) Ask a question — citations show like `(filename#chunkN)`")

# --------------------------
# File loading helpers
# --------------------------
def load_file(uploaded):
    """Return extracted text from PDF/DOCX/TXT."""
    name = uploaded.name
    data = uploaded.read()
    if name.lower().endswith(".pdf"):
        # Try pdfplumber first (better layout), fall back to pypdf
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text += t + "\n"
            return text
        except Exception:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
            except Exception:
                return ""
    elif name.lower().endswith(".docx"):
        import docx2txt, tempfile, os as _os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return docx2txt.process(tmp_path) or ""
        finally:
            try: _os.remove(tmp_path)
            except: pass
    else:
        # .txt or unknown → try utf-8
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def chunk_text(full_text, source_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([full_text], metadatas=[{"source": source_name}])
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = f"{source_name}#chunk{i+1}"
    return docs

# --------------------------
# Vector store (OpenAI embeddings)
# --------------------------
@st.cache_resource(show_spinner=False)
def build_vs(all_chunks):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(all_chunks, embeddings)

def citations_from(docs, max_refs=4):
    seen, out = set(), []
    for d in docs:
        tag = d.metadata.get("chunk_id") or d.metadata.get("source") or "doc"
        if tag not in seen:
            out.append(tag); seen.add(tag)
        if len(out) >= max_refs:
            break
    return ", ".join(out)

def answer_with_rag(vs, question):
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    context = ""
    for d in docs:
        tag = d.metadata.get("chunk_id") or d.metadata.get("source") or "doc"
        context += f"[{tag}]\n{d.page_content}\n\n"

    system_prompt = (
        "You are a Supplier Quality Auditor assistant. Use ONLY the provided context to answer. "
        "Cite sources as (filename#chunkN). If the answer is not in the context, say you don't know."
    )
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

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
# UI
# --------------------------
uploaded_files = st.file_uploader(
    "Upload one or more files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if "vs" not in st.session_state:
    st.session_state.vs = None

c1, c2 = st.columns([1,1])

with c1:
    if st.button("🔨 Build index (OpenAI)"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            all_chunks = []
            with st.spinner("Extracting and chunking..."):
                for f in uploaded_files:
                    txt = load_file(f)
                    if txt and txt.strip():
                        all_chunks.extend(chunk_text(txt, f.name))
            if not all_chunks:
                st.error("No text extracted from the files.")
            else:
                try:
                    with st.spinner("Embedding & indexing with OpenAI..."):
                        st.session_state.vs = build_vs(all_chunks)
                    st.success(f"Index ready. {len(all_chunks)} chunks.")
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

with c2:
    q = st.text_input("Ask an audit question… (e.g., “What are supplier PPAP requirements?”)")
    if q and st.session_state.vs:
        with st.spinner("Thinking…"):
            try:
                st.markdown(answer_with_rag(st.session_state.vs, q))
            except Exception as e:
                st.error(f"LLM call failed: {e}")
    elif q and not st.session_state.vs:
        st.info("Build the index first, then ask a question.")
