# 🧠 DocMind — RAG-Powered Document Q&A

A production-quality **Retrieval-Augmented Generation (RAG)** app built with LangChain, ChromaDB, and OpenAI. Upload any PDF, ask questions, get source-cited answers with full conversation memory.

---

## Architecture

```
INGESTION PIPELINE
──────────────────────────────────────────────────────────────
  PDF/TXT  →  Load  →  Chunk  →  Embed        →  ChromaDB
 (files)    PyPDF    RCTS*     ada-002 1536d    vector store

QUERY PIPELINE
──────────────────────────────────────────────────────────────
  Question  →  Embed  →  MMR Retrieve  →  LLM  →  Answer
  + History    ada-002    top-k chunks    GPT    + Sources

* RecursiveCharacterTextSplitter
```

| Component | Technology | Interview Talking Point |
|---|---|---|
| Document loading | PyPDFLoader, TextLoader | Preserves page metadata for citations |
| Chunking | RecursiveCharacterTextSplitter | Splits on paragraphs first, preserves semantics |
| Embeddings | text-embedding-ada-002 | 1536-dim vectors, semantic not keyword matching |
| Vector store | ChromaDB | Local, no server, cosine similarity search |
| Retrieval | MMR (k=3, fetch_k=9) | Diverse results, not 3 copies of same chunk |
| LLM | gpt-4o-mini, temp=0 | Deterministic, grounded, cheap |
| Memory | ConversationBufferMemory | Enables follow-up questions |
| UI | Streamlit + Chainlit | Both options included |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. API key
cp .env.example .env
# edit .env → OPENAI_API_KEY=sk-...

# 3. Run (pick one)
streamlit run app.py          # Streamlit UI (best for deployment)
chainlit run chainlit_app.py  # Chainlit UI (best chat UX)
```

---

## Deploy Free to Streamlit Cloud

1. Push to GitHub (public repo)
2. Go to share.streamlit.io → Connect repo → Deploy
3. Add `OPENAI_API_KEY` in the Secrets panel
4. Get a public URL — done

---

## F&Q

**"Walk me through your RAG pipeline"**
> "Two phases. Ingestion: I load documents with LangChain's PyPDFLoader, split them into 500-character chunks with 50-character overlap using RecursiveCharacterTextSplitter, embed each chunk with OpenAI's text-embedding-ada-002 to get 1536-dimensional vectors, and store those in ChromaDB locally. At query time: I embed the question with the same model, retrieve the top-3 most semantically similar chunks using MMR retrieval, and pass those chunks plus conversation history to GPT as context."

**"Why RAG instead of fine-tuning?"**
> "RAG is better when documents change — you just re-index, no retraining. Fine-tuning bakes knowledge into weights: expensive, slow, and static. RAG also provides source attribution which is critical for trust and auditability. Fine-tuning is better when you need the model to adopt a specific tone, style, or domain vocabulary."

**"What is chunking and why does chunk size matter?"**
> "Chunking splits documents to fit in LLM context windows and sharpen retrieval. Too small: a 50-character chunk has no context — a sentence divorced from its paragraph is ambiguous. Too large: a 2000-character chunk dilutes the relevance signal — the retrieved chunk may contain the answer buried in noise. I use RecursiveCharacterTextSplitter which splits on paragraph breaks first, then newlines, then sentences, preserving semantic coherence."

**"What are embeddings?"**
> "Dense numerical vectors encoding semantic meaning. text-embedding-ada-002 maps text to 1536-dimensional space. Semantically similar text clusters together — 'car' and 'automobile' have similar vectors. At query time, we compute cosine similarity between the question vector and all stored chunk vectors to find the most relevant chunks. This is fundamentally different from keyword search."

**"What is MMR retrieval?"**
> "Maximal Marginal Relevance balances relevance and diversity. Without it, if a document mentions 'neural networks' ten times in similar sentences, all three retrieved chunks might be nearly identical. MMR fetches nine candidates (fetch_k=9), then selects three that maximize relevance to the query while minimizing similarity to already-selected chunks."

**"How does conversation memory work here?"**
> "ConversationBufferMemory stores the full message history verbatim and injects it into each query. So 'what about the second point?' works because the LLM has context from previous turns. The tradeoff is token cost — long conversations get expensive. ConversationSummaryMemory solves this by summarizing old turns."

**"What would you improve?"**
> "Several things: add a reranker (cross-encoder) after MMR retrieval for better precision; add LangSmith tracing for observability; try hybrid search combining vector + BM25 keyword search; swap OpenAI embeddings for local sentence-transformers to reduce cost; add metadata filtering so users can search within a specific document."

---

## Project Structure

```
rag-project/
├── app.py              ← Streamlit UI (main entry point)
├── chainlit_app.py     ← Chainlit UI (alternative)
├── rag_engine.py       ← Core RAG logic (well-commented)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
