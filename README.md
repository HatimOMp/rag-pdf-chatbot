# 🤖 RAG PDF Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about any PDF document, powered by **Mistral AI** embeddings, **FAISS** vector search and a custom RAG pipeline deployed with **Streamlit**.

## 🎯 Project Overview

This project implements a full RAG pipeline from scratch — without LangChain or any high-level abstraction framework. Every component (chunking, embedding, vector search, prompt construction, conversation management) is built and controlled manually, giving full transparency over the retrieval and generation process.

**Why RAG?**
Large Language Models have a knowledge cutoff and no access to private documents. RAG solves this by retrieving relevant passages from a document at query time and injecting them into the prompt — giving the LLM accurate, document-grounded answers with source citations.

## 🏗️ Architecture
📄 PDF Document
↓
📝 Text Extraction (PyMuPDF)
└── Page-level extraction with metadata
↓
✂️ Chunking (PDFProcessor)
├── 500-word chunks with 50-word overlap
└── Overlap preserves context at chunk boundaries
↓
🔢 Embedding (Mistral AI — mistral-embed)
├── 1024-dimensional dense vectors
└── Batch processing to respect rate limits
↓
🗄️ Vector Index (FAISS — IndexFlatIP)
├── L2-normalized vectors for cosine similarity
└── Exact nearest-neighbor search
↓
🔍 Retrieval
└── Top-k most semantically similar chunks
↓
🧠 Generation (Mistral — mistral-small-latest)
├── System prompt with strict grounding rules
├── Retrieved context injected into prompt
├── Multi-turn conversation history
└── Page-level source citations
↓
💬 Answer + Sources

## 💡 Key Technical Decisions

**Why FAISS over a managed vector DB?**
FAISS runs fully locally with no external dependencies or costs. For a single-document chatbot, exact search (IndexFlatIP) is fast enough and more accurate than approximate methods.

**Why chunking overlap?**
A 50-word overlap between consecutive chunks prevents relevant sentences from being split across chunk boundaries, improving retrieval recall on edge cases.

**Why Mistral AI?**
- State-of-the-art French AI company — highly relevant for the French market
- `mistral-embed` produces high-quality 1024-dim embeddings optimized for retrieval
- `mistral-small-latest` offers an excellent quality/cost tradeoff for RAG

**Why no LangChain?**
Building the pipeline from scratch gives full control over chunking strategy, retrieval logic and prompt construction — and demonstrates deeper understanding of RAG internals than using a high-level abstraction.

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/HatimOMp/rag-pdf-chatbot
cd rag-pdf-chatbot
```

**2. Create a virtual environment**
```bash
conda create -n rag-env python=3.10
conda activate rag-env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the project root:
MISTRAL_API_KEY=your_mistral_api_key_here

Get your free API key at https://console.mistral.ai

**5. Launch the app**
```bash
streamlit run app.py
```

## 🗂️ Project Structure
rag-pdf-chatbot/
│
├── pdf_processor.py       # PDF extraction and chunking
├── vector_store.py        # Mistral embeddings + FAISS index
├── rag_engine.py          # RAG pipeline + conversation management
├── app.py                 # Streamlit interface
├── requirements.txt       # Dependencies
└── .env                   # API key (not committed)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Mistral AI (mistral-small-latest) |
| Embeddings | Mistral AI (mistral-embed, 1024-dim) |
| Vector Store | FAISS (IndexFlatIP) |
| PDF Processing | PyMuPDF (fitz) |
| Frontend | Streamlit |
| Environment | python-dotenv |

## 📊 Features

- **Upload any PDF** — research papers, contracts, reports, manuals
- **Semantic search** — finds relevant passages by meaning, not just keywords
- **Source citations** — every answer includes the page numbers used
- **Conversation memory** — maintains multi-turn dialogue context
- **Chunk inspector** — expandable view of retrieved context for transparency
- **Adjustable retrieval** — slider to control how many chunks are retrieved

## 🔮 Potential Improvements

- **Hybrid search** — combine dense (semantic) and sparse (BM25) retrieval for better recall
- **Re-ranking** — add a cross-encoder re-ranker to improve chunk selection precision
- **Multi-document support** — extend to query across multiple PDFs simultaneously
- **Streaming responses** — stream LLM output token by token for better UX
- **Evaluation** — add RAG evaluation metrics (faithfulness, answer relevancy) using RAGAS

## 👤 Author

**Hatim Omari** — [LinkedIn](https://www.linkedin.com/in/hatim-omari/) · [GitHub](https://github.com/HatimOMp)