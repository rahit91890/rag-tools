#!/usr/bin/env python3
"""
RAG Tools - Main Pipeline Module
Retrieval-Augmented Generation system for document search and question answering.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class RAGPipeline:
    """
    Main RAG pipeline class for document ingestion, embedding, and retrieval.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = './data/index'):
        """
        Initialize RAG pipeline with embedding model and vector store.
        
        Args:
            model_name: Name of the sentence-transformers model
            index_path: Path to store FAISS index and metadata
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        self.load_index()
    
    def load_index(self):
        """Load existing FAISS index and metadata from disk."""
        index_file = self.index_path / 'faiss.index'
        metadata_file = self.index_path / 'metadata.pkl'
        
        if index_file.exists() and metadata_file.exists():
            print("Loading existing index...")
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            print(f"Loaded {len(self.documents)} documents from index.")
        else:
            print("Creating new index...")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        index_file = self.index_path / 'faiss.index'
        metadata_file = self.index_path / 'metadata.pkl'
        
        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        print(f"Index saved with {len(self.documents)} documents.")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts for each document
        """
        if not documents:
            return
        
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{'id': i} for i in range(len(self.documents) - len(documents), len(self.documents))])
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents given a query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
        
        Returns:
            List of dicts containing documents, scores, and metadata
        """
        if len(self.documents) == 0:
            print("No documents in index. Please add documents first.")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'document': self.documents[idx],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity score
                    'distance': float(dist),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        return results
    
    def answer_question(self, question: str, top_k: int = 3, use_llm: bool = False) -> str:
        """
        Answer a question using retrieved documents.
        
        Args:
            question: Question to answer
            top_k: Number of documents to retrieve
            use_llm: Whether to use LLM (placeholder for now)
        
        Returns:
            Answer string
        """
        results = self.search(question, top_k=top_k)
        
        if not results:
            return "No relevant documents found to answer the question."
        
        # Simple concatenation of top results
        context = "\n\n".join([f"[{r['rank']}] {r['document']}" for r in results])
        
        if use_llm:
            # Placeholder for LLM integration
            answer = f"Based on the following context:\n\n{context}\n\nAnswer: [LLM integration coming soon. For now, review the context above to answer: {question}]"
        else:
            # Return mock answer with context
            answer = f"Question: {question}\n\nRelevant context found:\n{context}\n\n[Note: Install an LLM library like transformers or openai to generate automatic answers]"
        
        return answer
    
    def clear_index(self):
        """Clear all documents from the index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.metadata = []
        print("Index cleared.")


if __name__ == "__main__":
    # Quick test
    rag = RAGPipeline()
    
    # Sample documents
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
        "Natural language processing helps computers understand and generate human language.",
        "RAG combines retrieval and generation for more accurate question answering."
    ]
    
    rag.add_documents(sample_docs)
    rag.save_index()
    
    # Test search
    print("\n" + "="*60)
    query = "What is Python?"
    print(f"Query: {query}")
    results = rag.search(query, top_k=2)
    for r in results:
        print(f"\nRank {r['rank']} (score: {r['score']:.3f}):")
        print(r['document'])
    
    print("\n" + "="*60)
    print("RAG Pipeline initialized successfully!")
