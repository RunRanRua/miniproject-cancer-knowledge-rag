# Cancer Knowledge RAG

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
* Clinical trials
* Related information

The system retrieves relevant information from the knowledge base before generating an answer, allowing responses to be grounded in the underlying source documents.

---

## Architecture

```text
                    React + TypeScript
                           │
                      HTTP / SSE
                           │
                           ▼
                       FastAPI
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             Redis            RAG Service
          Session / Cache          │
                                   ▼
                              Retriever
                                   │
                                   ▼
                               ChromaDB
                                   │
                              Vector Search
                                   │
                                   ▼
                              LLM Provider
                              ┌────┴─────┐
                              │          │
                              ▼          ▼
                           Ollama    External API
```

### Main components

**React + TypeScript**

Provides the web interface for interacting with the RAG system.

**FastAPI**

Provides the backend REST API and connects the frontend with the RAG pipeline and storage components.

**LangChain**

Used for document processing, prompt construction, retrieval, and LLM integration.

**ChromaDB**

Acts as the vector store for the cancer knowledge base. Documents are embedded and stored together with relevant metadata for semantic retrieval.

**Redis**

Used for application state such as conversation history and response caching.

**Ollama**

Provides a local LLM runtime for development and personal use without requiring a paid external API.

**External LLM Providers**

The LLM layer is designed to be provider-independent so that users can optionally configure their own API credentials and use external model providers.

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

## Retrieval Pipeline

The core RAG pipeline is:

```text
NCI Resources
      │
      ▼
Document Loading
      │
      ▼
Text Cleaning
      │
      ▼
Document Splitting
      │
      ▼
Embedding
      │
      ▼
ChromaDB
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Context
      │
      ▼
LLM
      │
      ▼
Answer + Sources
```

The initial version focuses on semantic vector retrieval rather than hybrid keyword/vector search.

---

## LLM Providers

The project separates the RAG pipeline from the underlying LLM provider.

### Local development

The primary development setup uses **Ollama**:

```text
FastAPI
   │
   ▼
LLM Provider
   │
   ▼
Ollama
   │
   ▼
Local Model
```

This allows the project to be developed and tested locally without requiring a paid LLM API.

### External providers

The architecture also allows users to configure external LLM providers using their own API credentials.

For example:

```text
LLM_PROVIDER=ollama
```

or:

```text
LLM_PROVIDER=external
API_KEY=...
```

Provider-specific implementations are kept separate from the core RAG logic so that changing the model provider does not require rewriting the retrieval pipeline.

---

## Project Structure

```text
cancer-knowledge-rag/
│
├── frontend/                  # React + TypeScript
│
├── backend/                   # FastAPI application
│   ├── app/
│   │   ├── routes/
│   │   └── services/
│   │
│   ├── ingestion/
│   └── storage/
│
├── data/
│   └── sources.json
│
├── vector_store/              # Local vector store
│
├── evaluation/                # RAG evaluation
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

The project structure may evolve as additional components are implemented.

---

## Getting Started

### Requirements

* Python 3.12+
* Node.js
* npm
* Ollama
* Redis
* Git

### Backend

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start Redis.

Start Ollama and make sure the required model is available.

Then start the FastAPI server:

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI's interactive API documentation is available at:

```text
http://localhost:8000/docs
```

### Frontend

Install the frontend dependencies:

```bash
cd frontend
npm install
```

Start the development server:

```bash
npm run dev
```

---

## Configuration

Create a `.env` file based on `.env.example`.

Example local configuration:

```env
LLM_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=minimax-m3:cloud

REDIS_URL=redis://localhost:6379

CHROMA_PERSIST_DIRECTORY=./vector_store
```

API keys for external providers should never be committed to the repository.

---

## API

The backend exposes endpoints such as:

```text
POST /api/chat
POST /api/documents
GET  /api/health
```

Example chat request:

```json
{
  "session_id": "user_001",
  "message": "What are the symptoms of lung cancer?"
}
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "Lung Cancer",
      "source": "National Cancer Institute",
      "url": "https://www.cancer.gov/..."
    }
  ]
}
```

The API design may evolve during development.

---

## Development Roadmap

### Phase 1 — Knowledge Base

* [ ]  Identify relevant NCI pages
* [ ]  Implement document loading
* [ ]  Clean and normalize source documents
* [ ]  Split documents into chunks
* [ ]  Generate embeddings
* [ ]  Store documents in ChromaDB

### Phase 2 — RAG

* [ ]  Implement semantic retrieval
* [ ]  Build prompts
* [ ]  Integrate Ollama
* [ ]  Generate grounded answers
* [ ]  Return source metadata

### Phase 3 — Backend

* [ ]  Build FastAPI application
* [ ]  Implement `/chat`
* [ ]  Implement `/documents`
* [ ]  Add request validation
* [ ]  Add error handling

### Phase 4 — Conversation and Performance

* [ ]  Add Redis-based conversation history
* [ ]  Add response caching
* [ ]  Support session-based conversations

### Phase 5 — Frontend

* [ ]  Build React chat interface
* [ ]  Display retrieved sources
* [ ]  Add loading and error states
* [ ]  Support conversation sessions

### Phase 6 — Evaluation

* [ ]  Create a domain-specific evaluation dataset
* [ ]  Evaluate retrieval quality
* [ ]  Evaluate answer relevance
* [ ]  Evaluate source grounding
* [ ]  Compare different embedding models

### Phase 7 — Deployment

* [ ]  Dockerize the application
* [ ]  Configure production environment
* [ ]  Add external LLM provider support
* [ ]  Deploy backend and frontend

---

## Design Goals

The project focuses on several principles:

### Grounded generation

Answers should be based on retrieved source documents rather than relying solely on the model's internal knowledge.

### Source transparency

Relevant source information should be returned with generated answers whenever possible.

### Provider independence

The RAG pipeline should not depend on a single LLM provider.

### Local-first development

The system should be usable locally with Ollama and open-source/local components without requiring paid API access.

### Modular architecture

The retrieval, storage, API, frontend, and model layers should remain independently replaceable.

---

## Data and Attribution

This project uses publicly available information from the National Cancer Institute.

NCI is the primary source of the current knowledge base. Source URLs and attribution metadata are preserved where possible.

For information reuse and attribution policies, refer to the official NCI policies:

https://www.cancer.gov/policies/copyright-reuse

---

## Disclaimer

This project is an independent educational and technical project.

It is **not affiliated with, sponsored by, or endorsed by the National Cancer Institute (NCI)**.

The generated responses are intended for informational and educational purposes only. They are not a substitute for professional medical advice, diagnosis, or treatment.

Users should consult qualified healthcare professionals for medical decisions.
