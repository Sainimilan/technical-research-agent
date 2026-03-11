# ⚓ Data Harbor
### A Fully Offline GPU-Accelerated RAG System

> Upload documents. Ask questions. Get intelligent, cited answers — entirely on your local machine. No API keys. No internet. No data leaving your device.

---

## 🚀 What This Does

The **Data Harbor** is a production-grade Retrieval-Augmented Generation (RAG) system that lets you have a conversation with your documents. Upload one or more PDFs, ask questions in natural language, and get structured answers with source citations, page numbers, and confidence scores — all powered by a local LLM running on your own hardware.

---

## ✨ Features

### Core Intelligence
- 🧠 **Conversational Memory** — follows up on previous questions across the full session
- 📄 **Multi-PDF Support** — upload and query across multiple documents simultaneously
- 🔍 **Semantic Search** — finds relevant content by meaning, not just keywords
- ⚡ **Streaming Responses** — answers appear token by token, ChatGPT-style

### Retrieval Quality
- ✂️ **Semantic Chunking** — splits documents by topic shifts, not fixed character counts
- 🎯 **Cross-Encoder Reranking** — retrieves 10 candidates, reranks to top 3 for maximum accuracy
- 📖 **Page Citations** — every source shows the exact page number it came from
- 📊 **Real Confidence Scores** — actual cosine similarity scores, not fake percentages

### User Interface
- 💬 **Chat Interface** — full conversation history with message bubbles
- 🤖 **Model Selector** — switch between any locally installed Ollama model live
- 🗂️ **Document Library** — sidebar shows all uploaded docs with stats
- 📋 **Document Dashboard** — pages, chunks, word count shown after upload
- 🔎 **Source Cards** — expandable citations with filename, page, and relevance score

### Engineering
- 🔒 **Fully Offline** — zero external API calls, complete data privacy
- 🐳 **No Docker Required** — Qdrant runs from a local file
- 🏎️ **Batch Embedding** — 5-10x faster document processing
- 🛡️ **Full Error Handling** — clean messages if Ollama or Qdrant isn't running

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Streamlit UI  │  ← Chat interface, file upload, model selector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │  ← Orchestrates all modules
└────────┬────────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────┐              ┌──────────────┐
│ Embedder │              │  PDF Loader  │
│(MiniLM)  │              │ (page-aware) │
└────┬─────┘              └──────┬───────┘
     │                           │
     ▼                           ▼
┌──────────┐              ┌──────────────┐
│  Qdrant  │◄─────────────│   Chunker    │
│  Vector  │              │  (semantic)  │
│  Store   │              └──────────────┘
└────┬─────┘
     │  top-10 candidates
     ▼
┌──────────────┐
│ Cross-Encoder│  ← reranks to top 3
│  Reranker    │
└──────┬───────┘
       │  top-3 chunks + page numbers
       ▼
┌──────────────┐
│  Mistral LLM │  ← via Ollama (local)
│  via Ollama  │
└──────┬───────┘
       │
       ▼
  Structured Answer
  + Page Citations
  + Confidence Score
```

---

## 📁 Project Structure

```
data-harbor/
│
├── app.py                   # Streamlit UI — chat interface, upload, sidebar
├── requirements.txt         # Python dependencies
├── README.md
├── qdrant_storage/          # Local Qdrant database (auto-created)
│
└── rag_core/
    ├── pdf_loader.py        # Page-aware PDF text extraction + metadata
    ├── chunking.py          # Semantic chunking + fixed chunking fallback
    ├── embeddings.py        # SentenceTransformer embeddings (cached)
    ├── vector_store.py      # Qdrant operations + page number storage
    ├── reranker.py          # Cross-encoder reranking (NEW)
    ├── llm.py               # Ollama LLM + model listing
    └── rag_pipeline.py      # Central orchestrator
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.9+ |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| Vector Database | Qdrant (local file, no Docker) |
| LLM | Mistral / any model via Ollama |
| PDF Parsing | pypdf |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sainimilan/data-harbor.git
cd data-harbor
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama
Download from [ollama.com](https://ollama.com) then:
```bash
ollama serve
ollama pull mistral
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
qdrant-client
sentence-transformers
pypdf
requests
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | Optional | NVIDIA RTX (for faster LLM inference) |
| Storage | 5 GB free | 10 GB free |
| OS | Windows 10/11, Linux, macOS | Any |

---

## 🎮 How to Use

1. **Start the app** — `streamlit run app.py`
2. **Upload a PDF** — click the upload area and select one or more PDFs
3. **Process the document** — click the ⚙️ Process button and wait for the dashboard
4. **Ask a question** — type in the chat input at the bottom
5. **Choose your mode:**
   - **Fast** — concise direct answer
   - **Deep** — full structured report with Summary, Key Insights, Implementation Plan, Risks, and Conclusion
6. **View sources** — expand the 🔎 Sources section to see page citations
7. **Follow up** — ask follow-up questions, the system remembers the conversation
8. **Switch models** — use the sidebar to switch between Ollama models

---

## 🔍 Query Modes

### Fast Mode
Best for: quick lookups, simple factual questions
```
"What is data cleaning?"
"How many steps are in the data science lifecycle?"
```

### Deep Mode
Best for: complex topics, study notes, detailed explanations
```
"Explain the full data science lifecycle with all its stages"
"What are the risks and limitations of machine learning models?"
```

---

## 🧠 How RAG Works (Under the Hood)

```
1. PDF uploaded
        ↓
2. Text extracted page by page (page numbers preserved)
        ↓
3. Semantic chunking — splits on topic shifts, not character count
        ↓
4. Each chunk embedded into a 384-dim vector (all-MiniLM-L6-v2)
        ↓
5. Vectors + metadata (page_num, filename, doc_id) stored in Qdrant
        ↓
── At query time ──
        ↓
6. Query embedded into vector
        ↓
7. Top-10 candidates retrieved by cosine similarity
        ↓
8. Cross-encoder reranks to top-3 (reads query + chunk together)
        ↓
9. Context + chat history injected into LLM prompt
        ↓
10. Mistral generates structured answer
        ↓
11. Answer streamed to UI with page citations + confidence score
```


## 📄 License

MIT License — free to use, modify, and distribute.