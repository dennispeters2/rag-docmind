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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Circuit-brain SVG (light blue, used in sidebar)
BRAIN_SVG = """<svg width="48" height="48" viewBox="0 0 52 52" xmlns="http://www.w3.org/2000/svg">
  <line x1="26" y1="4" x2="26" y2="10" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="26" y1="10" x2="18" y2="10" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="48" y1="26" x2="42" y2="26" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="42" y1="26" x2="42" y2="34" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="26" y1="48" x2="26" y2="42" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="26" y1="42" x2="34" y2="42" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="4" y1="26" x2="10" y2="26" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <line x1="10" y1="26" x2="10" y2="18" stroke="#38bdf8" stroke-width="1.5" opacity="0.7"/>
  <circle cx="26" cy="4" r="2" fill="#7dd3fc"/><circle cx="18" cy="10" r="2" fill="#7dd3fc"/>
  <circle cx="48" cy="26" r="2" fill="#7dd3fc"/><circle cx="42" cy="34" r="2" fill="#7dd3fc"/>
  <circle cx="26" cy="48" r="2" fill="#7dd3fc"/><circle cx="34" cy="42" r="2" fill="#7dd3fc"/>
  <circle cx="4" cy="26" r="2" fill="#7dd3fc"/><circle cx="10" cy="18" r="2" fill="#7dd3fc"/>
  <path d="M26 13 C20 13 15 16 13 21 C10 21 8 23 8 26 C8 29 10 31 13 31 C13 36 17 40 22 41 L22 44 L30 44 L30 41 C35 40 39 36 39 31 C42 31 44 29 44 26 C44 23 42 21 39 21 C37 16 32 13 26 13Z"
    fill="rgba(56,189,248,0.08)" stroke="#38bdf8" stroke-width="1.8" stroke-linejoin="round"/>
  <line x1="26" y1="13" x2="26" y2="41" stroke="#38bdf8" stroke-width="1" stroke-dasharray="2,3" opacity="0.4"/>
  <path d="M14 22 C16 20 18 22 17 25" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M13 27 C15 25 18 27 17 30" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M15 32 C17 30 20 32 19 35" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M38 22 C36 20 34 22 35 25" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M39 27 C37 25 34 27 35 30" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <path d="M37 32 C35 30 32 32 33 35" fill="none" stroke="#7dd3fc" stroke-width="1.2" stroke-linecap="round"/>
  <circle cx="26" cy="27" r="3.5" fill="#38bdf8" opacity="0.2"/>
  <circle cx="26" cy="27" r="1.5" fill="#7dd3fc"/>
</svg>"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Hide the white Streamlit header bar ── */
header[data-testid="stHeader"] { background: #0b0b12 !important; border-bottom: 1px solid #1a1a28 !important; }
div[data-testid="stDecoration"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0b0b12; color: #f0eee8; }

/* ── HIGH CONTRAST global text ── */
p, span, div, label { color: #f0eee8; }
[data-testid="stWidgetLabel"] > div > p { color: #f0eee8 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #ddd8ce !important; }
/* Selectbox */
div[data-baseweb="select"] > div { background: #13131e !important; border-color: #2a2a40 !important; color: #f0eee8 !important; }
/* Slider numbers */
.stSlider > div { color: #f0eee8 !important; }

.brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #f0eee8;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-top: 0.4rem;
}
.tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #8888aa;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.pipeline-box {
    background: #13131e;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.73rem;
    color: #aaa8c0;
    line-height: 2.1;
    margin-bottom: 1.2rem;
}
.pipeline-box .hi { color: #60d0f8; }
.pipeline-box .label { color: #f0eee8; font-weight: 600; }
.source-card {
    background: #13131e;
    border: 1px solid #2a2a40;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.78rem;
}
.src-meta {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #60d0f8;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.src-text { color: #d8d4cc !important; line-height: 1.65; }
.chat-q {
    background: #181828;
    border: 1px solid #2a2a40;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    font-size: 0.88rem;
    color: #f0eee8;
}
.chat-a {
    background: #0f0f1c;
    border: 1px solid #222235;
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    font-size: 0.88rem;
    color: #f0eee8;
    line-height: 1.75;
}
.chat-label {
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #606088;
    margin-bottom: 0.3rem;
    font-family: 'DM Mono', monospace;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #38bdf8;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}
.indexed-bar {
    background: #071820;
    border: 1px solid #1a4060;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
    color: #60d0f8;
    margin-bottom: 1.2rem;
}
.stButton > button {
    background: #38bdf8 !important;
    color: #04060f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 7px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover { background: #7dd3fc !important; }
section[data-testid="stSidebar"] {
    background: #07070f !important;
    border-right: 1px solid #1a1a28 !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #13131e !important;
    border: 1px solid #2a2a40 !important;
    color: #f0eee8 !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
}
div[data-testid="stExpander"] {
    background: #13131e !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span { color: #c8d4e8 !important; }
[data-testid="stFileUploader"] {
    background: #13131e !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #c8c4bc !important; }
.empty-state { text-align: center; padding: 5rem 2rem; }
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #555570;
    margin-bottom: 0.4rem;
}
.empty-sub { font-size: 0.75rem; color: #404058; font-family: 'DM Mono', monospace; }
hr { border-color: #1a1a28 !important; }

/* ── Tooltip fix — dark bg, light text ── */
div[data-testid="stTooltipIcon"] { color: #60d0f8 !important; }
div[role="tooltip"],
div[data-testid="stTooltipContent"],
.stTooltipContent {
    background: #1e1e32 !important;
    border: 1px solid #38bdf8 !important;
    color: #f0eee8 !important;
    border-radius: 6px !important;
}
div[role="tooltip"] p,
div[role="tooltip"] span,
div[data-testid="stTooltipContent"] p,
div[data-testid="stTooltipContent"] span {
    color: #f0eee8 !important;
}
/* Popover that Streamlit uses for help text */
div[data-baseweb="popover"] div,
div[data-baseweb="tooltip"] div {
    background: #1e1e32 !important;
    color: #f0eee8 !important;
    border: 1px solid #2a2a50 !important;
}
div[data-baseweb="popover"] *,
div[data-baseweb="tooltip"] * {
    color: #f0eee8 !important;
    background: #1e1e32 !important;
}
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
    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;">{BRAIN_SVG}<div class="brand">DocMind</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline" style="margin-top:0.3rem;">RAG Engine · v1.0</div>', unsafe_allow_html=True)

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
st.markdown(f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:0.2rem;">{BRAIN_SVG}<div class="brand" style="font-size:2.2rem;">Document Intelligence</div></div>', unsafe_allow_html=True)
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
