# Mini Project: Cancer Knowledge RAG

A domain-specific Retrieval-Augmented Generation (RAG) system for retrieving and answering questions about cancer based on publicly available information from the **National Cancer Institute (NCI)**.

The project combines semantic retrieval, vector search, conversational context, and configurable LLM providers into a full-stack application.

> **Disclaimer:** This is an independent educational project. It is not affiliated with, sponsored by, or endorsed by the National Cancer Institute (NCI). The information provided by this system is intended for informational and educational purposes only and should not be considered medical advice, diagnosis, or treatment recommendations.

---

## Overview

The goal of this project is to build a reliable cancer information assistant grounded in a curated knowledge base rather than relying solely on the language model's internal knowledge.

The knowledge base is primarily built from publicly available NCI resources, including general cancer information and information organized by cancer type.

For each cancer type, relevant information may include:

* Description
* Symptoms and signs
* Causes and risk factors
* Diagnosis
* Treatment
* Prevention
* Screening
* Related information

The system retrieves relevant documents from the knowledge base before generating an answer, allowing responses to be grounded in the underlying source documents.

---

## Architecture

```text
                       Streamlit UI
                           │
                    Direct Function Calls
                           │
                           ▼
                       RAG Pipeline
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
            Question Rewriter   Retriever
                  │                 │
                  │                 ▼
                  │             ChromaDB
                  │                 │
                  │            Vector Store
                  │                 │
                  └────────┬────────┘
                           ▼
                      LLM Provider
                      ┌────┴─────┐
                      │          │
                      ▼          ▼
                   Ollama    External API
```

### Main Components

**Streamlit**

Provides a simple, interactive web interface for asking questions and viewing answers with sources. Direct function calls to backend eliminate the need for FastAPI.

**LangChain**

Used for document processing, prompt construction, retrieval, question rewriting, and LLM integration using LangChain Expression Language (LCEL) for composable chains.

**ChromaDB**

Acts as the vector store for the cancer knowledge base. Documents are embedded and stored together with relevant metadata for semantic retrieval.

**In-Memory Chat History**

Manages conversation state using LangChain's `InMemoryChatMessageHistory` for session-based context, replacing the need for Redis or external state management.

**Ollama**

Provides a local LLM runtime for development and personal use without requiring a paid external API.

**External LLM Providers**

The LLM layer is designed to be provider-independent so that users can optionally configure their own API credentials and use external model providers.

---

## Code Documentation

All backend modules include comprehensive docstrings and inline comments explaining:

### Backend Modules

**`backend/generation/`**

- `llm.py` - LLM provider initialization (Ollama, OpenAI, etc.)
- `prompt.py` - RAG prompt templates for constraining LLM responses
- `question_rewriter.py` - Rewrites follow-up questions to standalone form using conversation history

**`backend/retrieval/`**

- `embeddings.py` - Embedding provider initialization (Ollama embeddings)
- `retriever.py` - Document retrieval from ChromaDB vector store
- `history_aware.py` - Context-aware retrieval pipeline using conversation history
- `vector_store.py` - Vector store initialization and persistence

**`backend/ingestion/`**

- `loader.py` - Loads markdown files with metadata preservation
- `splitter.py` - Two-stage document splitting (markdown-aware + recursive character)

**`backend/memory/`**

- `chat_history.py` - Session-based conversation management:
  - `get_session_history()` - Get or create session history
  - `delete_session()` - Completely remove a conversation
  - `session_exists()` - Check if a session exists
  - Uses LangChain's `InMemoryChatMessageHistory` for local storage

**`backend/rag/`**

- `chain.py` - Main RAG orchestration pipeline

### Frontend Module

**`frontend/main.py`**

- Streamlit web interface with:
  - **Session Management** - Create, switch, and delete conversations (labeled "Conversation 1", "Conversation 2", etc.)
  - **Interactive Chat** - Chat input for questions and message history display
  - **Source Attribution** - Collapsible expander showing sources with metadata (title, cancer type, topic, link)
  - **Current Session Indicator** - Visual marker (●) showing active conversation
  - **Auto-Numbering** - Conversations automatically numbered based on creation order
  - **Styled UI** - Professional gradient header, hover effects, and responsive layout

Each file includes:

1. **Module-level docstring** - Purpose and key features
2. **Function docstrings** - Arguments, return types, and usage examples
3. **Inline comments** - Explanations of key architectural decisions and logic

---

## Getting Started

### Prerequisites

- Python 3.9+
- Ollama running locally (for LLM and embeddings)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cancer-knowledge-rag.git
cd cancer-knowledge-rag
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama with the required models:

```bash
ollama pull minimax-m3:cloud  # For the LLM
ollama pull nomic-embed-text  # For embeddings
ollama serve
```

4. Run the Streamlit app:

```bash
streamlit run frontend/main.py
```

The app will open at `http://localhost:8501`

## Key Design Decisions

### In-Memory Chat History

Session-based conversation management uses LangChain's `InMemoryChatMessageHistory`, reducing infrastructure complexity for a personal/educational project.

### Conversation Numbering

Conversations are automatically numbered based on creation order:

- Each new conversation gets a unique UUID internally (never shown to users)
- Conversations are numbered sequentially: "Conversation 1", "Conversation 2", etc.
- The number is determined by the position in the `known_sessions` list
- Deleting a conversation updates the numbers for subsequent conversations
- The current conversation is marked with a bullet point (●) in the sidebar

### Metadata Preservation

Documents maintain rich metadata throughout the pipeline:

- Original source information (URL, title, cancer type)
- Document structure (markdown headers)
- Language and file path

This metadata is displayed alongside answers for source attribution and user trust.

### History-Aware Question Rewriting

Follow-up questions like "What about side effects?" are automatically rewritten to standalone form ("What are the side effects of chemotherapy?") using the LLM before retrieval. This improves retrieval accuracy for multi-turn conversations.

---

## Knowledge Sources

The current version primarily uses publicly available information from:

**National Cancer Institute (NCI)**
https://www.cancer.gov/

The ingestion pipeline is designed to retrieve, process, and index source documents while preserving metadata such as:

```text
source
title
URL
cancer type
topic
```

This metadata is also used to provide source information alongside generated answers.

---

## Testing Strategy

### Unit Tests

All tests use local data and mock external services:

- **`test_question_rewriter.py`** - Tests question rewriting with MockLLM
- **`test_loader.py`** - Tests markdown loading with local files
- **`test_splitter.py`** - Tests document splitting logic
- **`test_chat_history.py`** - Tests session management
- **`test_embeddings.py`** - Tests embedding with mocked OllamaEmbeddings
- **`test_history_aware.py`** - Tests history formatting

### Mocking Strategy

External services are mocked to enable fast, reliable testing:

```python
# MockLLM inherits from langchain_core.language_models.LLM
class MockLLM(LLM):
    def _call(self, prompt: str, **kwargs):
        return "Mocked response"

# OllamaEmbeddings is patched in tests
@patch("backend.retrieval.embeddings.OllamaEmbeddings")
def test_embeddings(mock_ollama):
    mock_ollama.return_value.embed_query.return_value = [0.1, 0.2, ...]
```

This approach ensures tests run quickly and don't require Ollama or external APIs to be running.

---

## Development Workflow

### Adding New Documentation

1. Add docstrings following the existing format (module-level, function-level)
2. Include inline comments for complex logic
3. Update README.md if adding new modules or changing architecture

### Running Streamlit in Development

```bash
# Enable Streamlit debug mode
streamlit run frontend/main.py --logger.level=debug
```

### Debugging Failed Tests

```bash
# Run a single test with verbose output
pytest tests/retrieval/test_history_aware.py -v -s
```

---

## Summary

This project demonstrates:

- ✅ Clean, modular architecture with separated concerns (ingestion, retrieval, generation, memory)
- ✅ Comprehensive code documentation with docstrings and inline comments
- ✅ Robust testing with mocked external dependencies
- ✅ User-friendly interface with Streamlit
- ✅ Production-ready RAG pipeline with source attribution
- ✅ Session-based conversation management
- ✅ Easy to extend with new embedding/LLM providers
