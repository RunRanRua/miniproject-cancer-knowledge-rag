"""Document Splitting Module

Splits long documents into smaller chunks for better retrieval performance.
Uses two-stage splitting:
1. Markdown-aware splitting: Preserves document structure
2. Recursive character splitting: Creates fixed-size chunks with overlap

This ensures chunks have semantic meaning while fitting in context windows.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

# Markdown headers to use for structural splitting
# These become metadata fields on chunks
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),     # Main heading
    ("##", "header_2"),    # Subheading
    ("###", "header_3"),   # Sub-subheading
]

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for RAG retrieval.
    
    Two-stage process:
    1. Split by markdown headers to preserve document structure
    2. Split by character for fixed-size chunks with overlap
    
    Args:
        documents: List of LangChain Documents to split
        
    Returns:
        List of smaller Document chunks with:
        - Preserved metadata from original documents
        - Added header metadata (header_1, header_2, header_3)
        - Text content suitable for embedding and retrieval
        
    Chunk sizing:
    - chunk_size=1000: Target chunk size in characters
    - chunk_overlap=200: Overlap for context preservation
    """
    # Stage 1: Markdown-aware splitting
    # Preserves document structure and adds header information to metadata
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # Keep headers in content
    )

    # Stage 2: Recursive character splitting
    # Ensures chunks fit in context windows with some overlap
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for doc in documents:
        # First split by markdown structure
        md_chunks = md_splitter.split_text(doc.page_content)

        # Then split each markdown chunk into smaller pieces
        for md_chunk in md_chunks:
            recursive_chunks = recursive_splitter.split_text(md_chunk.page_content)

            for recursive_chunk in recursive_chunks:
                # Merge metadata from original doc and markdown chunk
                # This preserves source info while adding header context
                metadata = {**doc.metadata, **md_chunk.metadata}
                chunks.append(Document(page_content=recursive_chunk, metadata=metadata))

    return chunks
