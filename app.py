import streamlit as st
from rag_core.rag_pipeline import process_pdf, ask_question

st.set_page_config(page_title="Technical Research Agent", layout="wide")

st.title("📚 Technical & Coding Research Agent")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if st.button("Process Document"):
        process_pdf("temp.pdf")
        st.success("Document processed and stored!")

st.divider()

query = st.text_input("Ask a question about the document:")

mode = st.radio("Select Mode:", ["fast", "deep"])

retrieved_text = ""

if st.button("Generate Answer"):
    if query:
        with st.spinner("Generating answer..."):
            answer, retrieved_text, confidence = ask_question(query, mode)

        st.markdown("### 📌 Answer")
        st.markdown(answer)
        st.info(f"Confidence Score: {confidence:.0f}%")

        
    else:
        st.warning("Please enter a question.")

with st.expander("🔎 Retrieved Context"):
    if retrieved_text:
        st.write(retrieved_text)

