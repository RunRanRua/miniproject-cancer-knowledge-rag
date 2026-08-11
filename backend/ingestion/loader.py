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
    """"Load document metadata from the manifest file"""
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_documents(
        data_dir = NCI_DATA_DIR, 
        manifest_path = MANIFEST_FILE
    ) -> list[Document]:
    """
    Load Markdown files and attach metadata from the manifest..

    Args:
        data_dir (Path): The path to the data directory.
        manifest_path (Path): The path to the manifest file.
    """
    manifest = _load_manifest(manifest_path)
    # create a lookup table:
    metadata_map = { item["path"]: item for item in manifest["documents"] }

    documents= []
    source = data_dir.name
    for file_path in data_dir.rglob("*.md"):
        relative_path = file_path.relative_to(data_dir)
        relative_path_str = relative_path.as_posix()
        matching_path_str = source + "/" + relative_path_str

        if matching_path_str not in metadata_map:
            logger.warning(f"Warning: {matching_path_str} is not found in the manifest.")
            continue

        metadata = metadata_map[matching_path_str]
        content = file_path.read_text(encoding="utf-8")
        cancer_type = relative_path.parts[0]
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

