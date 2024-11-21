from langchain.text_splitter import CharacterTextSplitter


def get_splits(docs, chunk_size):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    splits = splitter.split_documents(docs)
    return splits
