from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

def split_documents(documents) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for doc in documents:
        md_chunks = md_splitter.split_text(doc.page_content)

        for md_chunk in md_chunks:
            recursive_chunks = recursive_splitter.split_text(md_chunk.page_content)

            for recursive_chunk in recursive_chunks:
                metadata = {**doc.metadata, **md_chunk.metadata}
                chunks.append(Document(page_content=recursive_chunk, metadata=metadata))

    return chunks
