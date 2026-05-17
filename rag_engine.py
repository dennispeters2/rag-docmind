"""
RAG Engine — Core Logic (Modern LangChain LCEL API)
Every architectural decision is an interview talking point.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _clean_metadata(meta: dict) -> dict:
    """
    ChromaDB requires all metadata values to be str, int, float, or bool.
    PDF loaders often include None or other types — this sanitizes them.
    """
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif v is None:
            cleaned[k] = ""
        else:
            cleaned[k] = str(v)
    return cleaned


class RAGEngine:
    """
    Full RAG pipeline using ChromaDB directly + LangChain LCEL for generation.

    Phase 1: ingest_documents() — load → chunk → embed → ChromaDB
    Phase 2: query()            — embed → retrieve → LLM → answer + sources
    """

    def __init__(self, api_key, model="gpt-4o-mini", chunk_size=500,
                 chunk_overlap=50, top_k=3, persist_directory="./chroma_db"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.model_name = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.persist_directory = persist_directory

        # INTERVIEW: temperature=0 → deterministic, factual output (no creative drift)
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

        # INTERVIEW: RecursiveCharacterTextSplitter tries separators in order:
        # \n\n (paragraphs) → \n (lines) → ". " (sentences) → " " (words)
        # Preserves semantic coherence vs arbitrary character-position splits
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Wipe old DB on each new engine init to avoid schema conflicts
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        # INTERVIEW: ChromaDB is a local vector store — no server needed.
        # Stores (text, embedding vector, metadata) per chunk.
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # INTERVIEW: OpenAI embedding function — SAME model must be used for
        # both ingestion and querying, otherwise vectors are in different spaces.
        # text-embedding-ada-002 = 1536-dimensional semantic vectors.
        self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

        self._chat_history = []

    def ingest_documents(self, file_paths: List[Tuple[str, str]]) -> Dict[str, int]:
        """Phase 1: Load → Chunk → Clean metadata → Embed → Store"""
        all_docs = []
        for temp_path, orig_name in file_paths:
            suffix = Path(temp_path).suffix.lower()
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(temp_path, encoding="utf-8")
            else:
                continue
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = orig_name
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError("No supported documents found (PDF, TXT, MD).")

        # INTERVIEW: chunk_overlap prevents losing info at chunk boundaries —
        # a sentence cut in half at a boundary appears in both adjacent chunks
        chunks = self.text_splitter.split_documents(all_docs)

        # Filter out empty chunks (can happen with scanned PDFs)
        chunks = [c for c in chunks if c.page_content.strip()]

        texts = [c.page_content for c in chunks]
        # CRITICAL: clean metadata — ChromaDB rejects None values and complex types
        metas = [_clean_metadata(c.metadata) for c in chunks]
        ids   = [f"chunk_{i}" for i in range(len(chunks))]

        # Batch insert (ChromaDB handles embedding via the embed_fn automatically)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

        self._chat_history = []
        return {"total_docs": len(file_paths), "total_chunks": len(chunks)}

    def query(self, question: str, chat_history=None) -> Dict[str, Any]:
        """Phase 2: Embed question → Retrieve chunks → Generate grounded answer"""
        count = self.collection.count()
        if count == 0:
            raise ValueError("No documents indexed. Call ingest_documents() first.")

        n = min(self.top_k, count)

        # INTERVIEW: ChromaDB embeds the question with the same embed_fn,
        # then finds the n most similar chunks by cosine similarity
        results = self.collection.query(
            query_texts=[question],
            n_results=n
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return {
                "answer": "I couldn't retrieve any relevant chunks. Try re-indexing your documents.",
                "sources": []
            }

        context = "\n\n---\n\n".join(docs)

        # INTERVIEW: System prompt constrains LLM to ONLY use retrieved context.
        # Without this, the model may answer from training data and bypass your documents.
        messages = [
            ("system", f"""You are a helpful document assistant. Use the context below to answer the question.
The context comes from the user's uploaded documents.
If the answer is in the context, answer it clearly and in full.
Only say you don't know if the topic is truly absent from the context.

Context from documents:
{context}""")
        ]

        # INTERVIEW: Conversation memory — last 3 exchanges kept in prompt.
        # Enables follow-up questions like "expand on the second point".
        for turn in self._chat_history[-6:]:
            messages.append((turn["role"], turn["content"]))
        messages.append(("human", "{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)

        # INTERVIEW: LCEL pipe — prompt | LLM | string output parser
        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question})

        # Save to memory
        self._chat_history.append({"role": "human",     "content": question})
        self._chat_history.append({"role": "assistant", "content": answer})

        # INTERVIEW: cosine distance → relevance score: 1 - distance
        # score of 1.0 = perfect match, 0.0 = completely unrelated
        sources = []
        for text, meta, dist in zip(docs, metas, distances):
            sources.append({
                "text":   text[:600] + ("..." if len(text) > 600 else ""),
                "source": meta.get("source_file", meta.get("source", "unknown")),
                "page":   meta.get("page", ""),
                "score":  round(1 - dist, 3)
            })

        return {"answer": answer, "sources": sources}

    def clear_history(self):
        self._chat_history = []

    def get_info(self) -> Dict[str, Any]:
        return {
            "status":               "loaded" if self.collection.count() > 0 else "empty",
            "total_vectors":        self.collection.count(),
            "embedding_model":      "text-embedding-ada-002",
            "embedding_dimensions": 1536,
            "similarity_metric":    "cosine",
            "chunk_size":           self.chunk_size,
            "chunk_overlap":        self.chunk_overlap,
            "top_k":                self.top_k,
        }
