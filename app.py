# app.py

import streamlit as st
from rag_core.rag_pipeline import process_uploaded_file, ask_question
from rag_core.llm import list_available_models
from rag_core.vector_store import get_collection_stats

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Technical Research Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    .confidence-high { color: #1E8449; font-weight: bold; font-size: 0.9em; }
    .confidence-mid  { color: #D35400; font-weight: bold; font-size: 0.9em; }
    .confidence-low  { color: #C0392B; font-weight: bold; font-size: 0.9em; }

    .source-card {
        background: #F8F9FA;
        border-left: 4px solid #2E86AB;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.85em;
        line-height: 1.5;
    }
    .source-header {
        display: flex;
        gap: 8px;
        margin-bottom: 6px;
        flex-wrap: wrap;
    }
    .badge {
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78em;
        font-weight: bold;
    }
    .badge-doc  { background: #1F4E79; color: white; }
    .badge-page { background: #1E8449; color: white; }
    .badge-score { background: #D35400; color: white; }

    .memory-indicator {
        font-size: 0.75em;
        color: #888;
        margin-top: 4px;
    }
    .doc-stat-card {
        background: #F0F4F8;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],        # [{role, content, sources, confidence}]
        "documents": {},       # {doc_id: {filename, num_chunks, num_pages, ...}}
        "active_doc_id": None, # None = search across all docs
        "selected_model": "mistral",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ── Helper Render Functions ──────────────────────────────────────────────────
# Defined FIRST so they are available everywhere below

def _render_confidence(confidence: float):
    """Colour-coded confidence gauge with progress bar."""
    if confidence >= 75:
        label = f"🟢 Confidence: {confidence:.1f}%"
        css = "confidence-high"
    elif confidence >= 45:
        label = f"🟠 Confidence: {confidence:.1f}%"
        css = "confidence-mid"
    else:
        label = f"🔴 Confidence: {confidence:.1f}%"
        css = "confidence-low"

    st.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)
    st.progress(min(int(confidence), 100))


def _render_sources(sources: list):
    """
    Render expandable source citations with:
    - Document filename badge
    - Page number badge  ← new in Phase 2
    - Relevance score badge
    - Chunk text preview
    """
    if not sources:
        return

    with st.expander(f"🔎 View {len(sources)} source(s) used"):
        for src in sources:
            filename = src.get("filename", "unknown")
            page_num = src.get("page_num", "?")
            score    = src.get("score", 0)
            text     = src.get("text", "")
            rerank   = src.get("rerank_score", None)

            rerank_str = (
                f'<span class="badge badge-score">Rerank: {rerank:.2f}</span>'
                if rerank is not None else ""
            )

            st.markdown(f"""
<div class="source-card">
  <div class="source-header">
    <span class="badge badge-doc">📄 {filename}</span>
    <span class="badge badge-page">📖 Page {page_num}</span>
    <span class="badge badge-score">🎯 {score:.1f}%</span>
    {rerank_str}
  </div>
  <div>{text[:450]}{'...' if len(text) > 450 else ''}</div>
</div>
""", unsafe_allow_html=True)


def _render_memory_indicator(chat_history: list):
    """Show how many previous turns the LLM has in memory."""
    turns = len([m for m in chat_history if m["role"] == "user"])
    if turns > 1:
        st.markdown(
            f'<p class="memory-indicator">🧠 Memory: {min(turns, 3)} '
            f'previous exchange(s) included in context</p>',
            unsafe_allow_html=True
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Model selector
    st.subheader("🤖 LLM Model")
    available_models = list_available_models()
    st.session_state.selected_model = st.selectbox(
        "Select model",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models else 0
        ),
        help="Models available in your local Ollama installation"
    )

    st.divider()

    # Mode selector
    st.subheader("🔍 Query Mode")
    mode = st.radio(
        "Analysis depth",
        ["fast", "deep"],
        captions=["Quick answer", "Structured report"],
        horizontal=True
    )

    st.divider()

    # Document library
    st.subheader("📂 Document Library")

    if st.session_state.documents:
        doc_options = {"🔍 Search all documents": None}
        for doc_id, meta in st.session_state.documents.items():
            label = f"📄 {meta['filename']}"
            doc_options[label] = doc_id

        selected_label = st.selectbox(
            "Search scope",
            options=list(doc_options.keys())
        )
        st.session_state.active_doc_id = doc_options[selected_label]

        # Per-document stats
        for doc_id, meta in st.session_state.documents.items():
            with st.expander(f"📄 {meta['filename']}"):
                c1, c2 = st.columns(2)
                c1.metric("Pages", meta.get("num_pages", "—"))
                c2.metric("Chunks", meta.get("num_chunks", "—"))
                st.caption(f"Words: {meta.get('word_count', 0):,}")
                if meta.get("author") and meta["author"] != "Unknown":
                    st.caption(f"Author: {meta['author']}")
                if meta.get("title") and meta["title"] != meta["filename"]:
                    st.caption(f"Title: {meta['title']}")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Conversation memory stats
    if st.session_state.messages:
        st.subheader("🧠 Conversation Memory")
        total_turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.caption(f"{total_turns} question(s) in this session")
        st.caption("Last 3 exchanges sent to LLM as context")

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗂️ Clear Docs", use_container_width=True):
            st.session_state.documents = {}
            st.session_state.active_doc_id = None
            st.rerun()

# ── Main Area ────────────────────────────────────────────────────────────────

st.title("📚 Technical Research Agent")
st.caption("Fully offline · GPU-accelerated · Multi-document · Semantic RAG")

# ── Document Upload ──────────────────────────────────────────────────────────

with st.expander(
    "📤 Upload Documents",
    expanded=len(st.session_state.documents) == 0
):
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs and query across all of them"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:

            # Skip already processed docs
            already_loaded = any(
                meta["filename"] == uploaded_file.name
                for meta in st.session_state.documents.values()
            )
            if already_loaded:
                st.info(f"✅ '{uploaded_file.name}' is already loaded.")
                continue

            if st.button(
                f"⚙️ Process '{uploaded_file.name}'",
                key=f"btn_{uploaded_file.name}"
            ):
                with st.status(
                    f"Processing '{uploaded_file.name}'...",
                    expanded=True
                ) as status:
                    try:
                        st.write("📖 Extracting text page by page...")
                        st.write("✂️ Applying semantic chunking...")
                        st.write("🧠 Generating embeddings in batch...")
                        st.write("💾 Storing in Qdrant with page metadata...")

                        result = process_uploaded_file(uploaded_file)

                        st.session_state.documents[result["doc_id"]] = {
                            "filename":   result["filename"],
                            "num_chunks": result["num_chunks"],
                            "num_pages":  result["num_pages"],
                            "word_count": result["word_count"],
                            "char_count": result["char_count"],
                            "title":      result.get("title", result["filename"]),
                            "author":     result.get("author", "Unknown")
                        }

                        status.update(
                            label=f"✅ '{uploaded_file.name}' ready!",
                            state="complete"
                        )

                        # Document dashboard
                        st.success(f"**{result['filename']}** processed!")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("📄 Pages",  result["num_pages"])
                        c2.metric("✂️ Chunks", result["num_chunks"])
                        c3.metric("📝 Words",  f"{result['word_count']:,}")
                        c4.metric("🔤 Chars",  f"{result['char_count']:,}")

                        if result.get("author", "Unknown") != "Unknown":
                            st.caption(f"Author: {result['author']}")
                        if result.get("title", result["filename"]) != result["filename"]:
                            st.caption(f"Title: {result['title']}")

                    except ConnectionError as e:
                        status.update(label="❌ Connection error", state="error")
                        st.error(str(e))
                    except ValueError as e:
                        status.update(label="❌ Processing error", state="error")
                        st.error(str(e))
                    except Exception as e:
                        status.update(label="❌ Unexpected error", state="error")
                        st.error(f"Unexpected error: {e}")

# ── Chat Interface ───────────────────────────────────────────────────────────

st.divider()

# Render full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Re-render sources and confidence for assistant messages
        if message["role"] == "assistant":
            if message.get("confidence"):
                _render_confidence(message["confidence"])
            if message.get("sources"):
                _render_sources(message["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    if not st.session_state.documents:
        st.warning("⚠️ Please upload and process a document first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = None

        try:
            with st.spinner("🔍 Retrieving & reranking context..."):
                result = ask_question(
                    query=prompt,
                    mode=mode,
                    model=st.session_state.selected_model,
                    doc_id=st.session_state.active_doc_id,
                    chat_history=st.session_state.messages[:-1]
                )

            # Stream answer token by token
            answer_text = result["answer"]
            streamed = ""
            for i in range(0, len(answer_text), 3):
                streamed = answer_text[:i + 3]
                message_placeholder.markdown(streamed + "▌")
            message_placeholder.markdown(answer_text)

            # Memory indicator
            _render_memory_indicator(st.session_state.messages)

            # Confidence + sources
            _render_confidence(result["confidence"])
            _render_sources(result["sources"])

        except ConnectionError as e:
            answer_text = str(e)
            message_placeholder.error(answer_text)
        except Exception as e:
            answer_text = f"❌ Error: {e}"
            message_placeholder.error(answer_text)

    # Save to history
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer_text,
        "sources":    result["sources"]    if result else [],
        "confidence": result["confidence"] if result else 0
    })