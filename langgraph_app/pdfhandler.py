from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


def read_pdf_to_docs(file):
    reader = PdfReader(file)
    docs = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        doc = Document(page_content=text, metadata={"page_number": idx + 1})
        docs.append(doc)
    return docs


def get_splits(docs, chunk_size):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    splits = splitter.split_documents(docs)
    return splits
