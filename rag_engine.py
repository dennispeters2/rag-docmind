"""
RAG Engine — Core Logic using FAISS vector store
FAISS installs cleanly on all platforms including Streamlit Cloud.
Every architectural decision is an interview talking point.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAGEngine:
    """
    Full RAG pipeline using FAISS + LangChain LCEL.

    Phase 1: ingest_documents() — load → chunk → embed → FAISS index
    Phase 2: query()            — embed → retrieve → LLM → answer + sources
    """

    def __init__(self, api_key, model="gpt-4o-mini", chunk_size=500,
                 chunk_overlap=50, top_k=3, persist_directory="./faiss_db"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.api_key = api_key
        self.model_name = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.persist_directory = persist_directory

        # INTERVIEW: temperature=0 → deterministic, factual output (no creative drift)
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

        # INTERVIEW: Same embedding model for ingestion AND querying — critical!
        # text-embedding-ada-002 = 1536-dimensional semantic vectors
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )

        # INTERVIEW: RecursiveCharacterTextSplitter tries separators in order:
        # \n\n (paragraphs) → \n (lines) → ". " (sentences) → " " (words)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore = None
        self._chat_history = []

    def ingest_documents(self, file_paths: List[Tuple[str, str]]) -> Dict[str, int]:
        """Phase 1: Load → Chunk → Embed → Store in FAISS"""
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
                # Clean metadata — ensure all values are strings
                for k, v in d.metadata.items():
                    if not isinstance(v, (str, int, float, bool)):
                        d.metadata[k] = str(v) if v is not None else ""
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError("No supported documents found (PDF, TXT, MD).")

        # INTERVIEW: chunk_overlap prevents losing info at chunk boundaries
        chunks = self.text_splitter.split_documents(all_docs)
        chunks = [c for c in chunks if c.page_content.strip()]

        # INTERVIEW: FAISS (Facebook AI Similarity Search) — fast, local, no server needed
        # Builds an index of embedding vectors for nearest-neighbor search
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        self._chat_history = []
        return {"total_docs": len(file_paths), "total_chunks": len(chunks)}

    def query(self, question: str, chat_history=None) -> Dict[str, Any]:
        """Phase 2: Embed question → Retrieve chunks → Generate grounded answer"""
        if not self.vectorstore:
            raise ValueError("No documents indexed. Call ingest_documents() first.")

        # INTERVIEW: similarity_search_with_score returns (doc, distance) pairs
        # FAISS uses L2 distance by default — lower score = more similar
        results = self.vectorstore.similarity_search_with_score(question, k=self.top_k)

        if not results:
            return {"answer": "No relevant content found in documents.", "sources": []}

        docs      = [r[0] for r in results]
        scores    = [r[1] for r in results]
        context   = "\n\n---\n\n".join(d.page_content for d in docs)

        # INTERVIEW: System prompt constrains LLM to ONLY use retrieved context
        messages = [
            ("system", f"""You are a helpful document assistant. Use the context below to answer the question.
The context comes from the user's uploaded documents.
If the answer is in the context, answer it clearly and in full.
Only say you don't know if the topic is truly absent from the context.

Context from documents:
{context}""")
        ]

        # INTERVIEW: Conversation memory — last 3 exchanges injected into prompt
        for turn in self._chat_history[-6:]:
            messages.append((turn["role"], turn["content"]))
        messages.append(("human", "{question}"))

        prompt = ChatPromptTemplate.from_messages(messages)

        # INTERVIEW: LCEL pipe — prompt | LLM | string output parser
        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question})

        self._chat_history.append({"role": "human",     "content": question})
        self._chat_history.append({"role": "assistant", "content": answer})

        # INTERVIEW: Convert L2 distance to 0-1 relevance score
        # Lower L2 distance = higher similarity, so we invert it
        sources = []
        for doc, score in zip(docs, scores):
            relevance = round(1 / (1 + score), 3)
            sources.append({
                "text":   doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""),
                "source": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
                "page":   doc.metadata.get("page", ""),
                "score":  relevance
            })

        return {"answer": answer, "sources": sources}

    def clear_history(self):
        self._chat_history = []

    def get_info(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "empty"}
        return {
            "status":               "loaded",
            "embedding_model":      "text-embedding-ada-002",
            "embedding_dimensions": 1536,
            "similarity_metric":    "L2 (FAISS)",
            "chunk_size":           self.chunk_size,
            "chunk_overlap":        self.chunk_overlap,
            "top_k":                self.top_k,
        }
