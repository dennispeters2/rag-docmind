"""
DocMind — Chainlit Version
Run: chainlit run chainlit_app.py
"""

import chainlit as cl
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()


@cl.on_chat_start
async def start():
    cl.user_session.set("rag_engine", None)
    await cl.Message(content="""# 🧠 DocMind — RAG Document Q&A

Upload a **PDF or text file** using the 📎 button below, then ask me anything about it.

**Pipeline:** PDF → Chunks → Embeddings (ada-002) → ChromaDB → MMR Retrieval → GPT → Answer + Sources
""").send()


@cl.on_message
async def message(msg: cl.Message):
    # File upload handling
    if msg.elements:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            await cl.Message(content="⚠️ Set OPENAI_API_KEY in your .env file.").send()
            return

        await cl.Message(content="⏳ Chunking, embedding, indexing...").send()
        try:
            engine = RAGEngine(api_key=api_key)
            paths = []
            for el in msg.elements:
                if hasattr(el, "path") and el.path:
                    paths.append((el.path, el.name))
            if paths:
                stats = engine.ingest_documents(paths)
                cl.user_session.set("rag_engine", engine)
                await cl.Message(content=f"✅ Indexed **{stats['total_chunks']} chunks** from **{stats['total_docs']}** file(s). Ask me anything!").send()
        except Exception as e:
            await cl.Message(content=f"❌ {e}").send()
        return

    engine: RAGEngine = cl.user_session.get("rag_engine")
    if not engine:
        await cl.Message(content="📎 Upload a document first.").send()
        return

    async with cl.Step(name="Retrieving relevant chunks"):
        result = engine.query(question=msg.content)

    sources = result["sources"]
    elements = []
    for i, src in enumerate(sources):
        pg = f" | Page {src['page']}" if src.get("page") is not None else ""
        elements.append(cl.Text(
            name=f"Chunk {i+1}: {src['source']}{pg} (score: {src['score']:.3f})",
            content=src["text"],
            display="side"
        ))

    await cl.Message(content=result["answer"], elements=elements).send()
