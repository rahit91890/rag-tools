# 🔍 RAG Tools

**Retrieval-Augmented Generation (RAG) Toolkit** - A powerful Python library for building intelligent document search and question-answering systems.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [Flask Web Interface](#flask-web-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Extension Guide](#extension-guide)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

RAG Tools is a comprehensive Python toolkit that combines **document retrieval** with **language generation** to create intelligent question-answering systems. It leverages state-of-the-art embedding models and efficient vector search to provide accurate, context-aware responses.

### What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that improves the quality of language model responses by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the query with retrieved context
3. **Generating** accurate answers based on the context

## ✨ Features

### Core Capabilities
- 📄 **Multi-Format Document Ingestion**: PDF, TXT, CSV, DOCX
- 🎯 **Semantic Search**: Vector-based similarity search using sentence-transformers
- ⚡ **Fast Retrieval**: FAISS-powered vector database for efficient search
- 🧠 **LLM Integration**: Ready for integration with OpenAI, Hugging Face, or custom models
- 🔄 **Persistent Storage**: Save and load your vector indices
- 🎨 **Dual Interface**: CLI for developers, Flask web UI for end users

### Technical Features
- Vector embeddings using open-source models (all-MiniLM-L6-v2)
- Efficient L2 distance-based similarity search
- Batch document processing with progress tracking
- Metadata support for enhanced filtering
- Extensible architecture for custom embeddings and retrievers

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│         Document Sources                 │
│    (PDF, TXT, CSV, DOCX, Web)           │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│      Ingestion Module                    │
│  • Text extraction                       │
│  • Chunking                              │
│  • Preprocessing                         │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│      Embedding Module                    │
│  • Sentence Transformers                 │
│  • Vector generation                     │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│      Vector Store (FAISS)                │
│  • Index storage                         │
│  • Similarity search                     │
└─────────────────┬────────────────────────┘
                  │
        User Query │
                  ▼
┌──────────────────────────────────────────┐
│      Retrieval Engine                    │
│  • Query embedding                       │
│  • Top-k retrieval                       │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│      Generation (Optional LLM)           │
│  • Context formatting                    │
│  • Answer generation                     │
└──────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Method 1: From Source

```bash
# Clone the repository
git clone https://github.com/rahit91890/rag-tools.git
cd rag-tools

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Direct Install

```bash
pip install sentence-transformers faiss-cpu numpy PyPDF2 Flask click
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from rag import RAGPipeline

# Initialize RAG system
rag = RAGPipeline()

# Add documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of AI.",
    "RAG combines retrieval with generation."
]
rag.add_documents(documents)
rag.save_index()

# Search
results = rag.search("What is Python?", top_k=2)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Document: {result['document']}\n")

# Question answering
answer = rag.answer_question("Explain RAG")
print(answer)
```

### 2. Run the Demo

```bash
python rag.py
```

## 📖 Usage

### CLI Interface

The command-line interface provides easy access to all RAG features:

```bash
# Index documents
python cli.py index --input ./documents --format pdf

# Search the index
python cli.py search "machine learning applications"

# Ask a question
python cli.py ask "What is neural network architecture?"

# Clear the index
python cli.py clear
```

### Flask Web Interface

Launch the web interface for interactive use:

```bash
python web_app.py
```

Then open your browser to `http://localhost:5000`

**Features:**
- Upload documents via drag-and-drop
- Real-time search
- Interactive Q&A
- Index management

### Python API

#### Document Ingestion

```python
from ingestion import DocumentLoader

# Load documents from various formats
loader = DocumentLoader()

# PDF
pdf_docs = loader.load_pdf("document.pdf")

# Text files
txt_docs = loader.load_txt("notes.txt")

# CSV
csv_docs = loader.load_csv("data.csv", text_column="content")

# Add to RAG
rag.add_documents(pdf_docs + txt_docs + csv_docs)
```

#### Advanced Search

```python
# Search with metadata filtering
results = rag.search(
    query="deep learning",
    top_k=10,
    metadata_filter={"category": "AI"}
)

# Get raw embeddings
embedding = rag.encoder.encode(["sample text"])
```

## 📁 Project Structure

```
rag-tools/
├── rag.py                 # Main RAG pipeline
├── cli.py                 # Command-line interface
├── web_app.py             # Flask web application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore            # Git ignore patterns
│
├── ingestion/            # Document ingestion modules
│   ├── __init__.py
│   ├── pdf_loader.py     # PDF processing
│   ├── txt_loader.py     # Text file processing
│   └── csv_loader.py     # CSV processing
│
├── embeddings/           # Embedding utilities
│   ├── __init__.py
│   └── encoder.py        # Embedding generation
│
├── examples/             # Example scripts
│   ├── basic_usage.py    # Simple examples
│   ├── advanced_rag.py   # Advanced features
│   └── custom_llm.py     # LLM integration
│
└── data/                 # Data storage (gitignored)
    ├── index/            # FAISS indices
    └── uploads/          # Uploaded documents
```

## ⚙️ Configuration

### Model Selection

Choose different embedding models:

```python
rag = RAGPipeline(
    model_name='all-mpnet-base-v2',  # More accurate but slower
    # model_name='all-MiniLM-L6-v2',   # Faster, good balance (default)
    # model_name='paraphrase-multilingual-MiniLM-L12-v2',  # Multilingual
)
```

### Index Configuration

```python
rag = RAGPipeline(
    index_path='./custom_index',
    embedding_dim=384  # Match your model's dimension
)
```

## 🔧 Extension Guide

### Adding Custom Document Loaders

```python
# ingestion/custom_loader.py
class CustomLoader:
    def load(self, filepath):
        # Your custom loading logic
        return list_of_documents

# Use in RAG
from ingestion.custom_loader import CustomLoader
loader = CustomLoader()
docs = loader.load("data.xyz")
rag.add_documents(docs)
```

### Integrating an LLM

```python
from transformers import pipeline

# Initialize LLM
llm = pipeline("text-generation", model="gpt2")

def custom_answer(rag, question):
    # Retrieve context
    results = rag.search(question, top_k=3)
    context = "\n".join([r['document'] for r in results])
    
    # Generate answer
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return llm(prompt, max_length=200)[0]['generated_text']
```

### Custom Embedding Models

```python
class CustomEmbedder:
    def encode(self, texts):
        # Your custom embedding logic
        return embeddings

rag.encoder = CustomEmbedder()
```

## 📚 Examples

### Example 1: Building a Documentation Search

```python
from rag import RAGPipeline
from ingestion import DocumentLoader

rag = RAGPipeline()
loader = DocumentLoader()

# Load all documentation
for pdf_file in glob.glob("docs/*.pdf"):
    docs = loader.load_pdf(pdf_file)
    rag.add_documents(docs, metadata=[{"source": pdf_file}])

rag.save_index()

# Search documentation
results = rag.search("How to install the library?")
```

### Example 2: Customer Support Bot

See `examples/advanced_rag.py` for a complete implementation.

### Example 3: Research Paper Analysis

See `examples/paper_analysis.py` for analyzing academic papers.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black rag.py ingestion/ embeddings/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Flask](https://flask.palletsprojects.com/) for web interface

## 📬 Contact

**Rahit Biswas**
- GitHub: [@rahit91890](https://github.com/rahit91890)
- Email: r.codaphics@gmail.com
- Website: [codaphics.com](https://codaphics.com)

---

⭐ **Star this repository** if you find it useful!

**Built with ❤️ by [Rahit Biswas](https://github.com/rahit91890)**
