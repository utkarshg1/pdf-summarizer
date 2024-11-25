import streamlit as st
import os
from langgraph_app.pdfhandler import read_pdf_to_docs
from langgraph_app.summarizer import get_app, get_summary
from langgraph_app.utils import get_splits

# Set Openai API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Set page config
st.set_page_config(page_title="PDF summarizer", page_icon="📝")

# Set page title
st.title("PDF Summarizer 📒🗒️📝")
st.subheader("by Utkarsh Gaikwad")

# File Uploader
uploaded_file = st.file_uploader("Upload a pdf", type="pdf")

if uploaded_file:
    docs = read_pdf_to_docs(uploaded_file)
    split_docs = get_splits(docs=docs, chunk_size=1000)
    st.write(f"Split docs length : {len(split_docs)}")
    with st.spinner():
        app = get_app()
        summary = get_summary(app, split_docs)
    st.write(summary)
