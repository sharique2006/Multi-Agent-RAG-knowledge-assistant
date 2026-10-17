# Personal Knowledge Assistant — Technical Deep Dive

## Complete Project Documentation for Interview Preparation

This document describes the Personal Knowledge Assistant: a multi-agent Retrieval-Augmented Generation (RAG) application that answers natural-language questions about uploaded documents with source citations. The project now includes the original Gradio interface and local RAG pipeline, plus a FastAPI REST API and PostgreSQL-backed API chat history.

> **Implementation status:** The local Gradio/RAG application is existing functionality. The FastAPI `/ask` endpoint and PostgreSQL history endpoints have been run and tested locally. AWS deployment, authentication, production hardening, and durable cloud file storage remain future work.

---

## Table of Contents

1. Project Overview
2. Technology Stack
3. Architecture Design
4. Multi-Agent System
5. RAG Pipeline
6. FastAPI REST API
7. PostgreSQL Chat History
8. Security and Deployment Considerations
9. Interview Questions and Answers
10. Technical Challenges and Solutions
11. Future Improvements

---

## 1. Project Overview

### What is this project?

A Personal Knowledge Assistant that allows users to:
- Upload documents such as PDF, DOCX, TXT, PNG, and JPG/JPEG.
- Ask natural-language questions about the content.
- Receive answers grounded in retrieved document chunks.
- Inspect citations that refer to source files and chunk IDs.
- Use a Gradio interface or call the RAG pipeline through REST endpoints.
- Save and retrieve API question-and-answer history from PostgreSQL.

### Why is it useful?

- **Specialized agents:** Retrieval, answer generation, and citation verification are separate components.
- **Grounded answers:** The generator is instructed to use only the retrieved context and to state when information is not found.
- **Traceable responses:** Citations refer back to source chunks.
- **Local vector search:** FAISS stores and searches embeddings locally.
- **API access:** FastAPI exposes the RAG pipeline to other clients.
- **Persistent API history:** PostgreSQL stores question-and-answer records beyond the lifetime of a single API request.

### Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | Gradio | Document upload, summary, Q&A, and chunk viewing |
| REST API | FastAPI, Uvicorn | Exposes the RAG pipeline to programmatic clients |
| LLM | OpenAI GPT-4o / GPT-4o-mini | Answer generation and supporting AI tasks |
| Orchestration | LangGraph | Controls the multi-agent workflow and conditional paths |
| Vector store | FAISS | Local similarity search over document embeddings |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text into vectors |
| Document parsing | pypdf, python-docx, Pillow | Extracts content from supported formats |
| Database | PostgreSQL | Stores API chat-history records |
| ORM / driver | SQLAlchemy, Psycopg 3 | Database models, sessions, and connectivity |
| Configuration | python-dotenv | Loads local environment variables |

---

## 2. Architecture Design

### High-level architecture

```text
                      User / API Client
                             |
                +------------+-------------+
                |                          |
          +-----v------+             +-----v------+
          | Gradio UI  |             | FastAPI    |
          | app.py     |             | api/main.py|
          +-----+------+             +-----+------+
                |                          |
                +------------+-------------+
                             |
                    LangGraph Orchestrator
                             |
             +---------------+----------------+
             |               |                |
        +----v-----+   +-----v------+   +-----v------+
        | Retriever|   | Generator  |   | Citation   |
        | Agent    |   | Agent      |   | Verifier   |
        +----+-----+   +------------+   +------------+
             |
        +----v-------+                 +----------------+
        | FAISS index|                 | PostgreSQL     |
        | + metadata |                 | Chat history   |
        +------------+                 +----------------+
```

### Document upload flow

```text
File -> Ingestion -> Chunking -> Embeddings -> FAISS Index
```

### Question-answer flow through the API

```text
HTTP Request
    -> FastAPI POST /ask
    -> Session-specific orchestrator
    -> Retrieve relevant FAISS chunks
    -> Generate answer from retrieved context
    -> Verify/format citations
    -> Save question and answer in PostgreSQL
    -> Return answer, citations, and history record ID
```

The existing Gradio interface uses the project's RAG components directly. The FastAPI API provides a separate programmatic entry point to the same RAG orchestration and a PostgreSQL history store.

---

## 3. Multi-Agent System

### Why use a multi-agent workflow?

A single LLM call can be difficult to debug and does not inherently validate whether an answer is grounded in retrieved documents. This project separates responsibilities:

- **Specialization:** Each agent has a defined task.
- **Modularity:** Agents can be changed independently.
- **Conditional control flow:** The graph can handle missing retrieval results and request a revision when verification is weak.
- **Traceability:** Logs and structured outputs help locate failures.

### Retriever Agent — `agents/retriever.py`

**Purpose:** Search for chunks relevant to the user's query.

The retriever uses the vector store and can apply query rewriting and metadata filtering. Retrieved chunks carry metadata such as chunk IDs and source details. The exact result depends on the query and the available index.

### Answer Generator Agent — `agents/answer_generator.py`

**Purpose:** Generate an answer from retrieved context.

The system prompt instructs the model to:
- Use only the supplied context.
- Avoid inventing facts not present in the context.
- State when the answer is not present in the uploaded documents.
- Include `[ChunkID: ...]` references for factual claims.
- Match the language style of the question.

The current configuration uses `Config.LARGE_MODEL` for answer generation.

### Citation Verifier Agent — `agents/citation_verifier.py`

**Purpose:** Verify and format the answer against retrieved chunks.

The verifier returns verification information and the orchestrator uses the result to decide whether to output the answer or attempt a revision. Verification improves traceability but should not be described as a mathematical guarantee that hallucinations are impossible.

### LangGraph Orchestrator — `agents/orchestrator.py`

The graph coordinates the pipeline:

1. Retrieve chunks.
2. If no chunks are available, create a no-results answer.
3. Generate an answer when chunks exist.
4. Verify citations and answer grounding.
5. Revise when the verification score indicates a retry is needed.
6. Format the final answer and citations.
7. Update the orchestrator's in-memory conversation memory.

The public method `AgentOrchestrator.query(query, filters=None)` returns a dictionary containing an answer and citations. Error results may include an `error` field.

---

## 4. RAG Pipeline

### Document ingestion — `rag/ingest.py`

Supported file formats in the application include:

| Format | Processing |
|---|---|
| PDF | Page-aware text extraction using pypdf |
| DOCX | Word document text extraction |
| TXT | Text file reading |
| PNG / JPG / JPEG | Image text extraction using the configured OpenAI vision model |

### Chunking — `rag/chunker.py`

The current configuration defines:
- Chunk size: approximately 400 tokens.
- Chunk overlap: approximately 50 tokens.

Chunking reduces the amount of text sent to the model and helps retrieval return focused passages. Overlap can preserve context near chunk boundaries.

### Vector store — `rag/vector_store.py`

The FAISS vector store:
- Loads `storage/vectors/index.faiss` when it exists.
- Loads chunk metadata from `storage/vectors/chunks.json`.
- Loads an embedding cache from `storage/vectors/embedding_cache.json`.
- Generates embeddings for uncached text using the configured embedding model.
- Normalizes vectors and searches the index for similar chunks.
- Saves index and metadata files locally.

The local index is not the same as the PostgreSQL database. FAISS stores searchable vectors and related local metadata; PostgreSQL stores API chat-history records.

### Conversation memory — `memory/conversation.py`

The orchestrator uses an in-memory conversation component to provide recent context to the generator. This memory is process-local and is not automatically persisted to PostgreSQL. The current API creates orchestrators by session ID in process memory; production multi-worker deployments need a more robust session/memory design.

---

## 5. FastAPI REST API

The REST API is implemented in `api/main.py` and runs locally with Uvicorn.

### Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | API welcome message and documentation path |
| `GET` | `/health` | Basic process health response |
| `POST` | `/ask` | Runs the existing RAG pipeline and saves the result to PostgreSQL |
| `POST` | `/history` | Manually saves a question-and-answer history record |
| `GET` | `/history/{session_id}` | Retrieves history for a session ID |

### `POST /ask`

Example request:

```json
{
  "session_id": "rag-test-1",
  "question": "Summarize the main points of the uploaded documents."
}
```

The endpoint:
1. Accepts a session ID and question.
2. Gets an orchestrator for that session, sharing the loaded vector store.
3. Calls `AgentOrchestrator.query()`.
4. Reads the answer and citation list from the pipeline result.
5. Saves the question and answer in PostgreSQL.
6. Returns the answer, citations, session ID, and database history ID.

Example response shape (illustrative only):

```json
{
  "session_id": "rag-test-1",
  "question": "Summarize the main points of the uploaded documents.",
  "answer": "Answer generated from the retrieved document context...",
  "citations": [],
  "history_id": 3
}
```

Actual answer text, citations, and record ID depend on the documents and database contents.

### `POST /history`

This endpoint is useful for manually testing database persistence.

Example request:

```json
{
  "session_id": "demo-session-1",
  "question": "What is RAG?",
  "answer": "RAG retrieves relevant information and uses it to generate a grounded answer."
}
```

A successful insert returns `201 Created` and the saved record, including its ID and creation timestamp.

### `GET /history/{session_id}`

Example:

```text
GET /history/rag-test-1
```

It returns saved records for that session, ordered by creation time and ID. The endpoint was tested locally and returned HTTP `200` with records from PostgreSQL.

### Local development commands

Run the Gradio UI in one terminal:

```powershell
python app.py
```

Run the FastAPI API in a separate terminal from the project root:

```powershell
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The `--reload` flag is for development. Use an appropriate production server configuration when deploying.

---

## 6. PostgreSQL Chat History

### Database layer — `api/db.py`

The database module:
- Loads environment variables from `.env`.
- Reads `DATABASE_URL`.
- Creates a SQLAlchemy engine.
- Defines `SessionLocal` for database sessions.
- Defines `Base` for declarative models.
- Provides `get_db()` as a FastAPI dependency that closes the session after the request.

The local connection URL uses the Psycopg 3 driver, for example:

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/rag_assistant
```

Replace the placeholder with your own database password. Do not commit real credentials.

### Data model — `api/models.py`

The `ChatHistory` table contains:

| Column | Purpose |
|---|---|
| `id` | Primary key |
| `session_id` | Groups history records for a session |
| `question` | Submitted question |
| `answer` | Saved answer |
| `created_at` | Timestamp generated by the database |

The table is created from SQLAlchemy metadata by the current application startup code. For a production system, use a database migration tool such as Alembic instead of relying on `create_all()` as the schema-evolution strategy.

### What PostgreSQL stores — and what it does not

**Stored:** API question-and-answer records for each session.

**Not automatically stored in PostgreSQL:** FAISS vectors, uploaded files, embedding cache, or the orchestrator's in-memory conversation context. These remain separate local-storage/in-memory components unless a later storage migration is implemented.

---

## 7. Configuration and Local Setup

### Prerequisites

- Python (3.10 or newer recommended for a clean environment).
- PostgreSQL installed and running.
- An OpenAI API key.
- Project dependencies installed from `requirements.txt`.

### Environment variables

Create `.env` from `.env.example` and configure:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/rag_assistant
```

Never commit `.env` or share real credentials in screenshots, logs, or public repositories. The repository's `.gitignore` excludes `.env`, virtual environments, local uploads, vector files, and local metadata.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Start the API

```powershell
python -m uvicorn api.main:app --reload
```

Check:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

The RAG API needs a valid OpenAI configuration and a usable FAISS index/chunk metadata set to answer document questions. A fresh clone does not automatically contain ignored local uploads or vector files.

---

## 8. Security and Deployment Considerations

The current implementation is suitable for local development and demonstration, but it should not be represented as fully production-hardened.

Before exposing it publicly, consider:

1. **Authentication and authorization:** The current endpoints accept a caller-supplied `session_id`; this is not user authentication.
2. **Session isolation:** Ensure a client cannot read another user's history by guessing a session ID.
3. **Conversation memory:** In-memory orchestrators are process-local and are not coordinated across multiple workers or instances.
4. **Database migrations:** Add Alembic or another migration strategy as the schema evolves.
5. **Input validation and rate limits:** Protect against oversized requests and API abuse.
6. **Error handling and logging:** Avoid leaking secrets or internal exception details to API clients.
7. **Persistent document storage:** The local uploads and FAISS index are ignored by Git. On cloud platforms with ephemeral filesystems, use persistent storage or S3 and implement loading/saving for those artifacts.
8. **Secrets management:** Use deployment environment variables or a managed secrets service.
9. **Health checks:** Extend health monitoring to check critical dependencies when needed.

### AWS deployment roadmap

A possible future deployment architecture is:
- **Compute:** EC2, Elastic Beanstalk, or ECS for the API and UI.
- **Database:** Amazon RDS for PostgreSQL.
- **File/object storage:** Amazon S3 for uploaded documents and durable vector artifacts.
- **Secrets:** Environment variables or a managed secrets service.
- **Monitoring:** Logs, metrics, and health checks.

**AWS deployment has not yet been completed as part of the locally tested steps described here.**

---

## 9. Interview Questions and Answers

### Q1. What is RAG?

**Answer:** Retrieval-Augmented Generation retrieves relevant passages from a knowledge base and supplies them as context to a language model. This project uses FAISS to retrieve document chunks, then generates an answer grounded in those chunks and includes citations.

### Q2. Why use a multi-agent architecture?

**Answer:** The system separates retrieval, answer generation, and citation verification. This makes the workflow easier to understand, debug, and improve than a single monolithic prompt. LangGraph coordinates the nodes and conditional transitions.

### Q3. How do you reduce hallucinations?

**Answer:** The generator is instructed to answer only from the retrieved context, include chunk citations, and explicitly state when the answer is not present. A verifier checks the answer and citations. These measures reduce unsupported answers but cannot guarantee that hallucinations never occur.

### Q4. Why use FAISS?

**Answer:** FAISS provides fast local vector similarity search without requiring a separate vector-database service. The project persists the index and related chunk metadata to local files. A cloud deployment must ensure these files are copied or stored durably.

### Q5. Why add FastAPI?

**Answer:** Gradio provides an interactive user interface, while FastAPI exposes the functionality through HTTP endpoints. Other applications can submit questions to `/ask` and retrieve saved history without interacting with the Gradio UI.

### Q6. What is the purpose of PostgreSQL?

**Answer:** PostgreSQL stores API chat-history records: session ID, question, answer, and creation timestamp. SQLAlchemy maps the Python `ChatHistory` model to the `chat_history` table, while Psycopg connects Python to PostgreSQL.

### Q7. Is the FAISS index stored in PostgreSQL?

**Answer:** No. The current FAISS index, chunks, and embedding cache are local files under `storage/vectors/`. PostgreSQL stores chat-history records. They are separate storage systems with different responsibilities.

### Q8. What happens in `POST /ask`?

**Answer:** FastAPI validates the request, obtains the orchestrator for the session, calls `AgentOrchestrator.query()`, collects the generated answer and citations, saves the question and answer in PostgreSQL, and returns the response with a history record ID.

### Q9. How did you test the API?

**Answer:** I started Uvicorn locally and used the automatically generated Swagger UI at `/docs`. I tested `POST /history` and `GET /history/{session_id}` to verify persistence and retrieval, then tested `POST /ask` and confirmed it returned HTTP 200 with a document-grounded answer and citations. The history endpoint returned the saved records for the session.

### Q10. Is the project deployed on AWS?

**Answer:** Not yet. The API and PostgreSQL history integration have been tested locally. AWS deployment is planned; it will require cloud database configuration, secure secrets, and durable storage for uploaded documents and FAISS artifacts.

### Q11. What would you improve before production?

**Answer:** Add authentication and authorization, isolate sessions securely, persist conversation memory appropriately, add database migrations and automated tests, improve rate limiting and observability, and use durable cloud storage for document and vector artifacts.

### Q12. How could the application scale?

**Answer:** Possible steps include using managed PostgreSQL, storing files in S3, separating the API from background document ingestion, using a shared persistent vector store or a managed vector database, adding authentication and rate limits, and moving session state out of process memory. These are future design options, not current implemented features.

---

## 10. Technical Challenges and Solutions

### Challenge 1: Keeping the existing RAG pipeline

**Approach:** Rather than rewriting retrieval and generation, the FastAPI `/ask` route calls the existing `AgentOrchestrator.query()` method. This preserves the project's existing LangGraph workflow.

### Challenge 2: Connecting PostgreSQL

**Approach:** A SQLAlchemy engine and session dependency provide database access. A `ChatHistory` model defines the table structure, and the history endpoints save and retrieve records.

### Challenge 3: Separating history from vector storage

**Approach:** FAISS continues to handle local similarity search. PostgreSQL stores chat history. This avoids confusing document-index persistence with conversation-record persistence.

### Challenge 4: Local data and cloud deployment

**Approach:** The current `.gitignore` excludes local uploads, vector files, and `.env`. That protects secrets and keeps generated data out of source control, but cloud deployment must separately provide the needed documents and vector artifacts.

---

## 11. Future Improvements

1. Add authentication and per-user authorization.
2. Add automated tests for API routes, database operations, and RAG error paths.
3. Use Alembic for database migrations.
4. Persist conversation memory and design safe session lifecycle handling.
5. Add document upload endpoints with validation and access control.
6. Add streaming answers if useful for the client experience.
7. Move uploaded files and durable vector artifacts to S3 for AWS hosting.
8. Add structured logging, monitoring, and rate limiting.
9. Improve hybrid retrieval by combining semantic and keyword search.
10. Add feedback collection and answer-quality evaluation.

---

## Conclusion

The Personal Knowledge Assistant combines a document-oriented RAG pipeline with a Gradio UI, LangGraph-controlled agents, FAISS vector search, citation verification, a FastAPI REST API, and PostgreSQL-backed API chat history.

The key distinction is that **the RAG API and PostgreSQL history have been tested locally, while cloud deployment and production hardening remain future work**. This accurately separates implemented functionality from planned improvements.
