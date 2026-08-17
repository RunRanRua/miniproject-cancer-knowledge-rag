"""Document Loading Module

Loads Markdown documents from the NCI data directory and attaches
metadata from a manifest file. This is the first step in the RAG pipeline.

The manifest file contains metadata for each document such as:
- title: Document title
- topic: Main topic (e.g., diagnosis, treatment)
- url: Source URL
"""

from pathlib import Path
import json
from langchain_core.documents import Document
from config import (
    NCI_DATA_DIR, 
    MANIFEST_FILE
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_manifest(manifest_path = MANIFEST_FILE) -> dict:
    """Load document metadata from the manifest file.
    
    Args:
        manifest_path: Path to the manifest JSON file
        
    Returns:
        Dictionary with 'documents' key containing list of metadata entries
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_documents(
        data_dir = NCI_DATA_DIR, 
        manifest_path = MANIFEST_FILE
    ) -> list[Document]:
    """Load Markdown files and attach metadata from the manifest.
    
    Process:
    1. Load manifest containing document metadata
    2. Build lookup table: file path -> metadata
    3. Iterate through all .md files in data directory
    4. Match files with manifest entries
    5. Create LangChain Document objects with metadata
    
    Args:
        data_dir: Path to the data directory (default: NCI_DATA_DIR)
        manifest_path: Path to the manifest file (default: MANIFEST_FILE)
        
    Returns:
        List of LangChain Documents with:
        - page_content: The markdown text
        - metadata: Source information (title, cancer_type, topic, url, etc.)
    """
    manifest = _load_manifest(manifest_path)
    # create a lookup table: file path -> metadata
    # This allows O(1) metadata lookup for each document
    metadata_map = { item["path"]: item for item in manifest["documents"] }

    documents= []
    source = data_dir.name  # e.g., "nci"
    
    # Recursively find all markdown files
    for file_path in data_dir.rglob("*.md"):
        relative_path = file_path.relative_to(data_dir)
        relative_path_str = relative_path.as_posix()
        matching_path_str = source + "/" + relative_path_str

        # Check if file exists in manifest
        if matching_path_str not in metadata_map:
            logger.warning(f"Warning: {matching_path_str} is not found in the manifest.")
            continue

        # Load metadata and content
        metadata = metadata_map[matching_path_str]
        content = file_path.read_text(encoding="utf-8")
        cancer_type = relative_path.parts[0]  # e.g., "breast_cancer"
        
        # Create Document with metadata preserved for retrieval
        document = Document(
            page_content=content,
            metadata={
                "source_id": source,
                "cancer_type": cancer_type,
                "title": metadata.get("title"),
                "topic": metadata.get("topic"),
                "subtopic": metadata.get("subtopic"),
                "url": metadata.get("url"),
                "file_path": matching_path_str,
                "language": "en",
            },
        )
        documents.append(document)
    return documents

