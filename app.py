# app.py

import streamlit as st
from rag_core.rag_pipeline import process_uploaded_file, ask_question
from rag_core.llm import list_available_models
from rag_core.vector_store import get_collection_stats

# ── Page Config ─────────────────────────────────────────────────────────────

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
    .confidence-high { color: #1E8449; font-weight: bold; }
    .confidence-mid  { color: #D35400; font-weight: bold; }
    .confidence-low  { color: #C0392B; font-weight: bold; }
    .source-card {
        background: #F2F3F4;
        border-left: 4px solid #2E86AB;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85em;
    }
    .doc-badge {
        background: #1F4E79;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)
# ── Helper Render Functions ──────────────────────────────────────────────────

def _render_confidence(confidence: float):
    """Render a colour-coded confidence gauge."""
    if confidence >= 75:
        label = f"🟢 Confidence: {confidence:.1f}%"
        color = "confidence-high"
    elif confidence >= 45:
        label = f"🟠 Confidence: {confidence:.1f}%"
        color = "confidence-mid"
    else:
        label = f"🔴 Confidence: {confidence:.1f}%"
        color = "confidence-low"

    st.markdown(f'<p class="{color}">{label}</p>', unsafe_allow_html=True)
    st.progress(int(confidence))


def _render_sources(sources: list, confidence: float):
    """Render expandable source citations with filename, chunk index and score."""
    if not sources:
        return
    with st.expander(f"🔎 View {len(sources)} source(s) used"):
        for i, src in enumerate(sources):
            st.markdown(f"""
<div class="source-card">
  <span class="doc-badge">{src.get('filename', 'unknown')}</span>
  Chunk #{src.get('chunk_index', '?')} &nbsp;|&nbsp;
  Relevance: <strong>{src.get('score', 0):.1f}%</strong><br><br>
  {src.get('text', '')[:400]}{'...' if len(src.get('text','')) > 400 else ''}
</div>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],           # chat history [{role, content}]
        "documents": {},          # {doc_id: {filename, num_chunks, num_pages, ...}}
        "active_doc_id": None,    # None = search across all docs
        "selected_model": "mistral",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    # Model selector — pulls live list from Ollama
    st.subheader("🤖 LLM Model")
    available_models = list_available_models()
    st.session_state.selected_model = st.selectbox(
        "Select model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models else 0,
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

        # Show stats for each document
        for doc_id, meta in st.session_state.documents.items():
            with st.expander(f"📄 {meta['filename']}"):
                col1, col2 = st.columns(2)
                col1.metric("Pages", meta.get("num_pages", "—"))
                col2.metric("Chunks", meta.get("num_chunks", "—"))
                st.caption(f"Words: {meta.get('word_count', '—'):,}")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Area ────────────────────────────────────────────────────────────────

st.title("📚 Technical Research Agent")
st.caption("Fully offline · GPU-accelerated · Multi-document RAG")

# ── Document Upload ──────────────────────────────────────────────────────────

with st.expander("📤 Upload Documents", expanded=len(st.session_state.documents) == 0):

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs and query across all of them"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip if already processed
            already_loaded = any(
                meta["filename"] == uploaded_file.name
                for meta in st.session_state.documents.values()
            )
            if already_loaded:
                st.info(f"✅ '{uploaded_file.name}' is already loaded.")
                continue

            if st.button(f"⚙️ Process '{uploaded_file.name}'", key=f"btn_{uploaded_file.name}"):
                with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status:
                    try:
                        st.write("📖 Extracting text...")
                        result = process_uploaded_file(uploaded_file)

                        st.write(f"✂️ Created {result['num_chunks']} chunks...")
                        st.write("🧠 Generating embeddings...")
                        st.write("💾 Storing in vector database...")

                        # Save to document library in session state
                        st.session_state.documents[result["doc_id"]] = {
                            "filename": result["filename"],
                            "num_chunks": result["num_chunks"],
                            "num_pages": result["num_pages"],
                            "word_count": result["word_count"],
                            "char_count": result["char_count"]
                        }

                        status.update(label=f"✅ '{uploaded_file.name}' ready!", state="complete")

                        # Show document dashboard
                        st.success(f"**{result['filename']}** processed successfully!")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("📄 Pages", result["num_pages"])
                        c2.metric("✂️ Chunks", result["num_chunks"])
                        c3.metric("📝 Words", f"{result['word_count']:,}")
                        c4.metric("🔤 Characters", f"{result['char_count']:,}")

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

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Re-render sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            _render_sources(message["sources"], message.get("confidence", 0))

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    if not st.session_state.documents:
        st.warning("⚠️ Please upload and process a document first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_answer = ""

        try:
            with st.spinner("Retrieving context..."):
                result = ask_question(
                    query=prompt,
                    mode=mode,
                    model=st.session_state.selected_model,
                    doc_id=st.session_state.active_doc_id,
                    chat_history=st.session_state.messages[:-1]  # exclude current user msg
                )

            # Stream the answer token by token
            answer_text = result["answer"]
            for i in range(0, len(answer_text), 3):
                full_answer = answer_text[:i+3]
                message_placeholder.markdown(full_answer + "▌")
            message_placeholder.markdown(answer_text)
            full_answer = answer_text

            # Show confidence score
            confidence = result["confidence"]
            _render_confidence(confidence)

            # Show sources
            _render_sources(result["sources"], confidence)

        except ConnectionError as e:
            full_answer = str(e)
            message_placeholder.error(full_answer)
        except Exception as e:
            full_answer = f"❌ Error: {e}"
            message_placeholder.error(full_answer)

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": result.get("sources", []) if "result" in dir() else [],
        "confidence": result.get("confidence", 0) if "result" in dir() else 0
    })

