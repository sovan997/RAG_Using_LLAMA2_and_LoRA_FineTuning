import streamlit as st
from rag_engine import load_documents, create_retriever, answer_query

st.set_page_config(page_title="RAG App with LLoRA-LLaMA2")
st.title("📄 RAG App using Finetuned LLaMA2 + Streamlit")

uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
query = st.text_input("Ask a question from the documents:")

if uploaded_files and query:
    with st.spinner("Processing documents and generating answer..."):
        docs = load_documents(uploaded_files)
        retriever = create_retriever(docs)
        response = answer_query(query, retriever)
        st.success(response)