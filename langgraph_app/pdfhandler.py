from pypdf import PdfReader
from langchain.docstore.document import Document


def read_pdf_to_docs(file):
    reader = PdfReader(file)
    docs = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        doc = Document(page_content=text, metadata={"page_number": idx + 1})
        docs.append(doc)
    return docs
