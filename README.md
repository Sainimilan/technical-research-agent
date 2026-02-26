
# 📚 Technical & Coding Research Agent

### Fully Offline GPU-Accelerated RAG System

## 🚀 Overview

The **Technical & Coding Research Agent** is a fully offline Retrieval-Augmented Generation (RAG) system that performs intelligent analysis over user-uploaded PDF documents.

It combines:

* 📄 PDF text extraction
* ✂️ Context-aware chunking
* 🧠 Semantic embeddings
* 📦 Vector similarity search (Qdrant)
* 🤖 Local LLM inference (Mistral via Ollama)
* 🌐 Interactive Streamlit UI

The entire system runs **locally**, with **no external API calls**, ensuring privacy, low cost, and GPU acceleration.

---

## 🧠 Architecture

```
User Query
    ↓
Embed Query
    ↓
Vector Similarity Search (Top-K)
    ↓
Context Retrieval
    ↓
Prompt Construction
    ↓
Local LLM (Mistral via Ollama)
    ↓
Structured Answer + Confidence Score
```

### Project Structure

```
rag-project/
│
├── app.py                     # Streamlit UI
├── requirements.txt
├── README.md
│
├── rag_core/
│   ├── pdf_loader.py          # PDF extraction
│   ├── chunking.py            # Text chunking logic
│   ├── embeddings.py          # SentenceTransformer embeddings
│   ├── vector_store.py        # Qdrant operations
│   ├── llm.py                 # Ollama LLM interface
│   └── rag_pipeline.py        # RAG orchestration
```

This modular structure ensures clear separation of concerns and maintainability.

---

## 🛠️ Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| UI              | Streamlit                               |
| Embeddings      | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | Qdrant (Docker)                         |
| LLM             | Mistral (via Ollama, local GPU)         |
| Language        | Python                                  |
| Deployment      | Fully Local                             |

---

## ✨ Features

* ✅ Upload and analyze any PDF
* ✅ Smart text chunking with overlap
* ✅ Semantic vector search (Top-K retrieval)
* ✅ Fast Mode (concise answers)
* ✅ Deep Mode (structured technical analysis)
* ✅ Confidence score based on retrieval
* ✅ Retrieved context transparency
* ✅ Fully offline, GPU-accelerated inference

---

## 🔐 Why Fully Offline?

* No API cost
* No external data exposure
* Works without internet
* Faster inference on GPU
* Suitable for private/enterprise environments

---

## ⚙️ Setup Instructions

### 1️⃣ Install Requirements

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Start Qdrant (Vector Database)

Make sure Docker Desktop is running.

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Verify:

```
http://localhost:6333
```

---

### 3️⃣ Install & Run Ollama (Local LLM)

Download Ollama:
[https://ollama.com](https://ollama.com)

Pull Mistral model:

```bash
ollama pull mistral
```

Test:

```bash
ollama run mistral
```

Then exit using:

```
/bye
```

---

### 4️⃣ Run Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. User uploads PDF
2. Text is extracted and chunked
3. Each chunk is embedded into vectors
4. Vectors are stored in Qdrant
5. Query is embedded
6. Top-K similar chunks retrieved
7. Context injected into prompt
8. Mistral generates structured answer

---

## 📊 Confidence Score

The system computes a confidence score based on the number of relevant chunks retrieved relative to the expected Top-K results.

Future improvements may use similarity scores for more granular confidence estimation.

---

## 🎯 Use Cases

* Technical document analysis
* Coding documentation summarization
* Research paper breakdown
* Structured learning roadmap extraction
* Internal enterprise document Q&A

---

## 🔮 Future Improvements

* Multi-PDF support
* Conversational memory
* Similarity-based confidence scoring
* Cloud deployment option
* Authentication layer

---

## 🏆 Unique Selling Points

* Fully offline RAG pipeline
* GPU-accelerated local inference
* Transparent retrieval mechanism
* Modular clean architecture
* No dependency on external APIs

---

## 📜 License

This project is built for academic and research demonstration purposes.

---

