"""
rag_retriever.py — RAG Knowledge Base for Coaching Agent
Extracted and adapted from trial.ipynb.
Manages the ChromaDB vector store and retrieval for clinical context.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class CoachingKnowledgeBase:
    """
    Manages the vector store for clinical/exercise knowledge retrieval.
    Built on your existing ChromaDB + HuggingFace embeddings setup.
    """

    def __init__(
        self,
        data_dir: str = "dataset/clean",
        persist_dir: str = "./chroma_coaching_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"Loading embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vectordb: Optional[Chroma] = None

    # ── Build or Load ────────────────────────────────────────────────────────

    def load_or_build(self, force_rebuild: bool = False) -> "CoachingKnowledgeBase":
        """
        Load existing ChromaDB if it exists, otherwise build from dataset/clean.
        Set force_rebuild=True to re-index from scratch.
        """
        if not force_rebuild and os.path.exists(self.persist_dir):
            print(f"Loading existing vector DB from: {self.persist_dir}")
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            count = self.vectordb._collection.count()
            print(f"Loaded {count} document chunks")
        else:
            print("Building vector DB from dataset...")
            documents = self._load_all_documents()
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
            print(f"Built DB with {len(documents)} chunks → saved to {self.persist_dir}")
        
        return self

    # ── Document Loading (from your notebook) ───────────────────────────────

    def _load_all_documents(self) -> List[Document]:
        """Load all .txt and .html files from dataset/clean directory."""
        all_docs = []

        # Load TXT files
        txt_files = list(self.data_dir.glob("**/*.txt"))
        for f in txt_files:
            all_docs.extend(self._load_txt(f))

        # Load HTML files
        html_files = list(self.data_dir.glob("**/*.html")) + list(self.data_dir.glob("**/*.htm"))
        for f in html_files:
            all_docs.extend(self._load_html(f))

        print(f"\nTotal chunks loaded: {len(all_docs)}")
        print(f"  TXT files: {len(txt_files)}, HTML files: {len(html_files)}")
        return all_docs

    def _load_txt(self, file_path: Path) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.text_splitter.split_text(content)
            return [
                Document(
                    page_content=chunk,
                    metadata={"source": "txt", "file": file_path.name, "chunk_id": i},
                )
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
            return []

    def _load_html(self, file_path: Path) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            if BS4_AVAILABLE:
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks_text if chunk)
            else:
                # Fallback: strip tags naively
                import re
                text = re.sub(r"<[^>]+>", " ", html_content)
                text = " ".join(text.split())

            chunks = self.text_splitter.split_text(text)
            return [
                Document(
                    page_content=chunk,
                    metadata={"source": "html", "file": file_path.name, "chunk_id": i},
                )
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
            return []

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> Tuple[str, List[str]]:
        """
        Retrieve relevant clinical context for a given query.
        
        Returns:
            (formatted_context_string, list_of_source_filenames)
        """
        if self.vectordb is None:
            raise RuntimeError("Knowledge base not loaded. Call load_or_build() first.")

        docs = self.vectordb.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant clinical guidance found.", []

        # Format retrieved docs into a readable block
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file", "unknown")
            sources.append(source)
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n".join(context_parts), list(set(sources))

    def get_retriever(self, k: int = 3):
        """Return a LangChain retriever object (for chain composition)."""
        if self.vectordb is None:
            raise RuntimeError("Knowledge base not loaded. Call load_or_build() first.")
        return self.vectordb.as_retriever(search_kwargs={"k": k})
