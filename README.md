# 🧠 Personal Knowledge Assistant

A multi-agent Retrieval-Augmented Generation (RAG) application for asking natural-language questions about your documents. It combines a Gradio interface, a FastAPI REST API, OpenAI models, LangGraph orchestration, FAISS retrieval, citation verification, and PostgreSQL-backed API chat history.

> **Project status:** The Gradio application and local RAG pipeline are part of the existing project. The FastAPI `/ask` endpoint and PostgreSQL chat-history endpoints have been implemented and tested locally. Cloud deployment is a future step, not a claim that the app is already deployed.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![OpenAI](https://img.shields.io/badge/AI-OpenAI-green.svg)
![Database](https://img.shields.io/badge/Database-PostgreSQL-4169E1.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Multi-format document support:** PDF, DOCX, TXT, PNG, and JPG/JPEG, using the existing document-ingestion pipeline.
- **Multi-agent RAG workflow:** Retrieval, answer generation, and citation verification coordinated with LangGraph.
- **Grounded answers:** The generator is instructed to answer from retrieved document context and cite relevant chunks.
- **FAISS vector search:** Local vector index, chunk metadata, and embedding cache.
- **Gradio interface:** Upload documents, generate summaries, ask questions, and inspect chunks.
- **FastAPI REST API:** Programmatic access to the RAG pipeline and chat history.
- **PostgreSQL chat history:** Store and retrieve question-and-answer records by session ID.
- **Configuration through environment variables:** API keys and database connection details are kept out of source code.
- **Local-first development:** Run the UI, API, FAISS index, and PostgreSQL locally.

## 🏗️ Architecture

```text
                 ┌──────────────────────────────┐
                 │        User / Client         │
                 └──────────────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
           ┌────────▼────────┐     ┌────────▼────────┐
           │  Gradio UI      │     │  FastAPI REST   │
           │  app.py         │     │  api/main.py    │
           └────────┬────────┘     └───────┬─────────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ LangGraph Orchestrator│
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
      ┌──────▼──────┐  ┌───────▼────────┐  ┌──────▼─────────┐
      │  Retriever  │  │ Answer Generator│  │ Citation       │
      │    Agent    │  │   (OpenAI)      │  │ Verifier       │
      └──────┬──────┘  └────────────────┘  └────────────────┘
             │
      ┌──────▼────────┐                    ┌─────────────────┐
      │ FAISS Vector  │                    │ PostgreSQL      │
      │ Store + Cache │                    │ Chat History    │
      └───────────────┘                    └─────────────────┘
```

### Request flow

1. Documents are ingested, split into chunks, embedded, and indexed in FAISS.
2. A user submits a question through Gradio or the REST API.
3. The orchestrator retrieves relevant chunks and generates an answer using the existing OpenAI-backed agents.
4. The citation verifier checks and formats the answer with source references.
5. Requests sent to `POST /ask` are saved in PostgreSQL along with the generated answer.

## 🧰 Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| User interface | Gradio |
| REST API | FastAPI, Uvicorn |
| Agent orchestration | LangGraph |
| LLM and embeddings | OpenAI API |
| Vector search | FAISS, NumPy |
| Database | PostgreSQL |
| ORM / database driver | SQLAlchemy, Psycopg 3 |
| Document processing | pypdf, python-docx, Pillow |
| Configuration | python-dotenv |

## 📁 Project Structure

```text
Rag_Full/
├── app.py                      # Gradio application
├── requirements.txt            # Python dependencies
├── .env.example                # Environment-variable template
├── .gitignore                  # Excludes secrets and local data
├── README.md                   # Project documentation
├── PROJECT_DETAILS.md           # Additional project details
├── api/
│   ├── main.py                 # FastAPI routes
│   ├── db.py                   # PostgreSQL engine and sessions
│   └── models.py               # SQLAlchemy database models
├── agents/
│   ├── retriever.py            # Retrieves relevant chunks
│   ├── answer_generator.py     # Generates grounded answers
│   ├── citation_verifier.py    # Verifies and formats citations
│   └── orchestrator.py         # LangGraph workflow
├── rag/
│   ├── ingest.py               # Document ingestion
│   ├── chunker.py              # Text chunking
│   └── vector_store.py         # FAISS index and embedding cache
├── memory/
│   └── conversation.py         # In-memory conversation context
├── utils/
│   ├── config.py               # Configuration and model settings
│   └── logger.py               # Logging
└── storage/
    ├── uploads/                # Local uploaded documents
    ├── vectors/                # Local FAISS index and cache
    └── metadata.json           # Local document metadata
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer recommended
- PostgreSQL 17 (or a compatible PostgreSQL version)
- An OpenAI API key
- Git (optional)

### 1. Clone or open the project

```bash
git clone <your-repository-url>
cd Rag_Full
```

If you already have the project locally, open a terminal in the project root.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable directly from `.venv`, or follow your organization's execution-policy guidance.

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
notepad .env
```

**macOS/Linux**
```bash
cp .env.example .env
```

Set the values required by the project:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/rag_assistant
```

Replace the example values with your own credentials. Do not add quotes unless your environment setup requires them. Never commit `.env` or paste real secrets into public issues or chat messages.

### 5. Create the PostgreSQL database

Create a database named `rag_assistant` using pgAdmin or `psql`. For example, when connected to PostgreSQL as an administrator:

```sql
CREATE DATABASE rag_assistant;
```

Make sure `DATABASE_URL` in `.env` points to the correct database and credentials. The API initializes the SQLAlchemy model tables when it starts.

### 6. Run the Gradio application

In one terminal, run:

```bash
python app.py
```

Open the local URL printed by Gradio, typically:

```text
http://127.0.0.1:7860
```

Use the UI to upload a supported document before asking document-specific questions.

### 7. Run the FastAPI service

Open a **second terminal** in the project root, activate the same virtual environment, and run:

```bash
python -m uvicorn api.main:app --reload
```

The API is available locally at:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

Keep the terminal running while you use the API.

## 🔌 REST API

### `GET /`

Returns a basic API welcome message and the docs path.

### `GET /health`

Returns a basic health response. This checks that the API process responds; it is not a full dependency-health check for OpenAI or PostgreSQL.

### `POST /ask`

Runs the existing RAG pipeline and saves the resulting question and answer to PostgreSQL.

**Request**
```json
{
  "session_id": "demo-session-1",
  "question": "Summarize the main points of the uploaded documents."
}
```

An optional `filters` object can be included if supported by the retriever's metadata filters.

**Response shape**
```json
{
  "session_id": "demo-session-1",
  "question": "Summarize the main points of the uploaded documents.",
  "answer": "Answer generated from the retrieved document context...",
  "citations": [],
  "history_id": 1
}
```

The answer, citation list, and history ID depend on the actual documents and execution result. The sample above illustrates the response shape only.

### `POST /history`

Manually stores a question-and-answer record. This endpoint is useful for testing persistence.

**Request**
```json
{
  "session_id": "demo-session-1",
  "question": "What is RAG?",
  "answer": "RAG retrieves relevant information and uses it to generate a grounded answer."
}
```

A successful insert returns `201 Created`.

### `GET /history/{session_id}`

Returns the saved question-and-answer records for the supplied session ID, ordered by creation time.

Example:

```text
GET http://127.0.0.1:8000/history/demo-session-1
```

### Test through Swagger UI

1. Open http://127.0.0.1:8000/docs.
2. Expand `POST /ask`.
3. Click **Try it out**, enter the JSON request, and click **Execute**.
4. Check the returned answer and citations.
5. Call `GET /history/{session_id}` with the same session ID to confirm that the answer was saved.

> **Note:** The API currently uses local FAISS files and in-process conversation memory. For a production deployment, review session isolation, authentication, request limits, database migrations, logging, and persistent storage.

## ⚙️ Configuration

The application configuration is managed in `utils/config.py`.

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | `400` | Approximate tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Maximum chunks retrieved |
| `LARGE_MODEL` | `gpt-4o` | Answer generation |
| `SMALL_MODEL` | `gpt-4o-mini` | Supporting AI tasks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Text embeddings |

These are the defaults currently defined in the project. Changing a model or retrieval setting may affect quality, latency, and API cost.

## 🔐 Security and Data Notes

- `.env` is listed in `.gitignore`; keep API keys and database credentials out of version control.
- The local `storage/uploads/` and `storage/vectors/` data are ignored by Git. A fresh clone will not automatically contain your existing documents or FAISS index.
- Do not expose the development server publicly as-is. Add authentication, authorization, rate limiting, and production-grade error handling before public use.
- The example API currently identifies history using a caller-provided `session_id`; it does not establish user identity or provide access control.
- The in-memory orchestrator/session map is process-local and is not suitable for coordinated multi-worker production use without redesign.
- For cloud hosting, use managed secrets and persistent object/block storage for uploaded documents and vector artifacts. A local disk on an ephemeral host should not be treated as durable storage.

## ☁️ Deployment Roadmap

The application has been developed and tested locally. Cloud deployment is a separate step.

A possible AWS architecture is:

- **Compute:** EC2, Elastic Beanstalk, or ECS for the API and UI.
- **Database:** Amazon RDS for PostgreSQL.
- **Object storage:** Amazon S3 for uploaded documents and any vector artifacts that need durable storage.
- **Secrets:** Environment configuration or a managed secret service.
- **Monitoring:** Application logs, health checks, and error alerts.

Before deployment, confirm that the selected hosting service supports the required Python packages and FAISS dependencies, configure production environment variables, and test document persistence after restarts.

## 🧪 Troubleshooting

| Problem | What to check |
|---|---|
| `OPENAI_API_KEY not found` | Confirm `.env` exists in the project root and contains a valid key. |
| `DATABASE_URL is missing` | Add `DATABASE_URL` to `.env` and restart the API. |
| PostgreSQL connection fails | Verify the PostgreSQL service, database name, username, password, port, and connection URL. |
| `No documents uploaded` | Confirm the FAISS index and chunk metadata exist, or upload documents through the Gradio UI. |
| `404 Not Found` for `/docs` | Open `http://127.0.0.1:8000/docs` exactly and confirm Uvicorn is running on port `8000`. |
| Import/module errors | Activate the intended virtual environment and run `python -m pip install -r requirements.txt`. |
| Empty or weak answer | Check that relevant documents were indexed and that retrieval returns useful chunks. |
| API works but history is empty | Use the same `session_id` for the history request and inspect the API response for database errors. |

## 🤝 Contributing

Contributions and improvements are welcome.

1. Create a feature branch.
2. Make a focused change.
3. Test the Gradio UI and relevant API endpoints.
4. Commit your changes without secrets, local uploads, or generated vector files.
5. Open a pull request describing the change and how it was tested.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

Built with [OpenAI](https://openai.com/), [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [FastAPI](https://fastapi.tiangolo.com/), [PostgreSQL](https://www.postgresql.org/), [SQLAlchemy](https://www.sqlalchemy.org/), and [Gradio](https://www.gradio.app/).
