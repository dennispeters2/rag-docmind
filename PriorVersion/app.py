"""
DocMind — RAG-Powered Document Q&A
Streamlit App — run with: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0b0b12; color: #e8e4dc; }

.brand {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #e8e4dc;
    letter-spacing: -0.03em;
    line-height: 1;
}
.tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #555568;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.pipeline-box {
    background: #111118;
    border: 1px solid #222230;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.72rem;
    color: #888899;
    line-height: 2;
    margin-bottom: 1.2rem;
}
.pipeline-box .hi { color: #b8f040; }
.pipeline-box .label { color: #e8e4dc; font-weight: 500; }
.source-card {
    background: #111118;
    border: 1px solid #222230;
    border-left: 3px solid #b8f040;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.76rem;
}
.src-meta {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #b8f040;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.src-text { color: #b0aca4; line-height: 1.65; }
.chat-q {
    background: #161622;
    border: 1px solid #222230;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.87rem;
    color: #e8e4dc;
}
.chat-a {
    background: #0d0d14;
    border: 1px solid #1a1a26;
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.87rem;
    color: #e8e4dc;
    line-height: 1.7;
}
.chat-label {
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #444455;
    margin-bottom: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #b8f040;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.indexed-bar {
    background: #0d1a04;
    border: 1px solid #2a4410;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
    color: #b8f040;
    margin-bottom: 1.2rem;
}
.stButton > button {
    background: #b8f040 !important;
    color: #080810 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 7px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover { background: #c8ff50 !important; }
section[data-testid="stSidebar"] {
    background: #080810 !important;
    border-right: 1px solid #16161f !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111118 !important;
    border: 1px solid #222230 !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
}
div[data-testid="stExpander"] {
    background: #111118 !important;
    border: 1px solid #222230 !important;
    border-radius: 8px !important;
}
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #2a2a38;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #383848;
    margin-bottom: 0.4rem;
}
.empty-sub {
    font-size: 0.75rem;
    color: #2a2a38;
    font-family: 'DM Mono', monospace;
}
hr { border-color: #16161f !important; }
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [
    ("rag_engine", None),
    ("chat_history", []),
    ("docs_loaded", False),
    ("doc_names", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🧠 DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">RAG Engine · v1.0</div>', unsafe_allow_html=True)

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])

    st.markdown("---")
    st.markdown("**Chunking**")
    chunk_size = st.slider("Chunk size (chars)", 200, 1500, 500, 50,
        help="Larger = more context per chunk, less precise retrieval")
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 300, 50, 10,
        help="Overlap prevents info loss at chunk boundaries")
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 3,
        help="More chunks = richer context, more tokens used")

    st.markdown("---")
    st.markdown("""<div class="pipeline-box">
<span class="label">① Load</span> <span class="hi">PyPDFLoader</span><br>
<span class="hi">↓</span><br>
<span class="label">② Chunk</span> <span>RecursiveCharacterSplitter</span><br>
<span class="hi">↓</span><br>
<span class="label">③ Embed</span> <span class="hi">text-embedding-ada-002</span><br>
<span class="hi">↓</span><br>
<span class="label">④ Store</span> <span>ChromaDB (local)</span><br>
<span class="hi">↓</span><br>
<span class="label">⑤ Retrieve</span> <span>MMR top-k</span><br>
<span class="hi">↓</span><br>
<span class="label">⑥ Generate</span> <span class="hi">GPT + sources</span>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Reset All"):
            for k in ["rag_engine", "chat_history", "docs_loaded", "doc_names"]:
                st.session_state[k] = [] if k in ["chat_history", "doc_names"] else None if k == "rag_engine" else False
            st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="brand" style="font-size:2rem;">Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Retrieval-Augmented Generation · Source-grounded answers</div>', unsafe_allow_html=True)

# Upload row
col_upload, col_btn = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded and api_key:
        if st.button("⚡ Index Documents"):
            with st.spinner("Chunking → Embedding → Storing in ChromaDB..."):
                try:
                    engine = RAGEngine(
                        api_key=api_key, model=model,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
                    )
                    temp_paths = []
                    for f in uploaded:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                            tmp.write(f.read())
                            temp_paths.append((tmp.name, f.name))

                    stats = engine.ingest_documents(temp_paths)
                    st.session_state.rag_engine = engine
                    st.session_state.docs_loaded = True
                    st.session_state.doc_names = [f.name for f in uploaded]
                    st.session_state.chat_history = []

                    for path, _ in temp_paths:
                        os.unlink(path)

                    st.success(f"✅ {stats['total_chunks']} chunks indexed from {stats['total_docs']} file(s)")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif not api_key and uploaded:
        st.warning("← Add API key")

# Indexed status bar
if st.session_state.docs_loaded:
    files = " · ".join(st.session_state.doc_names)
    st.markdown(f'<div class="indexed-bar"><span class="status-dot"></span>INDEXED: {files}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Chat ───────────────────────────────────────────────────────────────────────
if st.session_state.docs_loaded:
    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-q"><div class="chat-label">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-a"><div class="chat-label">DocMind</div>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source chunk(s) retrieved — click to inspect"):
                    for i, src in enumerate(msg["sources"]):
                        pg = f" · Page {src['page']}" if src.get("page") is not None else ""
                        st.markdown(f"""<div class="source-card">
<div class="src-meta">Chunk {i+1} · {src['source']}{pg} · relevance: {src['score']:.3f}</div>
<div class="src-text">{src['text']}</div>
</div>""", unsafe_allow_html=True)

    # Input form
    with st.form("qform", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input("question", placeholder="What are the main findings? Summarize section 2...", label_visibility="collapsed")
        with c2:
            ask = st.form_submit_button("Ask →")

    if ask and q.strip():
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.spinner("Retrieving → Generating..."):
            try:
                result = st.session_state.rag_engine.query(question=q)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant", "content": f"⚠️ {e}", "sources": []
                })
        st.rerun()

else:
    st.markdown("""<div class="empty-state">
<div class="empty-icon">📄</div>
<div class="empty-title">No documents indexed</div>
<div class="empty-sub">Upload PDFs above · Add your OpenAI key · Click Index Documents</div>
</div>""", unsafe_allow_html=True)
