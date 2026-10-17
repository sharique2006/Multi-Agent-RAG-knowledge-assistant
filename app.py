"""
Personal Knowledge Assistant - Multi-Agent RAG Application

A production-ready RAG application with a clean step-by-step UI:
1. Upload Documents
2. Get Document Summary
3. Ask Questions (Q&A)
4. View Document Chunks

Compatible with local running and Hugging Face Spaces.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from typing import List, Tuple, Dict, Optional
import json
import traceback

from utils.config import Config
from utils.logger import logger
from rag.ingest import DocumentIngestor
from rag.chunker import TextChunker
from rag.vector_store import VectorStore
from agents.orchestrator import AgentOrchestrator
from api.db import SessionLocal
from api.models import ChatHistory

# ============================================================================
# INITIALIZATION
# ============================================================================
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def initialize_system():
    """Initialize all system components."""
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
        return True
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"WARNING: {e}")
        print("Please ensure OPENAI_API_KEY is set in your .env file")
        return False


# Initialize on startup
system_ready = initialize_system()

# Global instances - initialized lazily to avoid startup errors
_ingestor = None
_chunker = None
_vector_store = None
_orchestrator = None

# Store current document info for session
_current_document = None
_current_chunks = []


def get_ingestor():
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor


def get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = TextChunker()
    return _chunker


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(get_vector_store())
    return _orchestrator


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def update_metadata(document_info: dict):
    """Update metadata file with new document info."""
    metadata_file = Config.METADATA_FILE
    
    try:
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {"documents": []}
        
        existing_hashes = {d.get("hash") for d in metadata["documents"]}
        if document_info.get("hash") not in existing_hashes:
            metadata["documents"].append(document_info)
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")


def get_uploaded_documents() -> List[Dict]:
    """Get list of uploaded documents."""
    metadata_file = Config.METADATA_FILE
    
    try:
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return metadata.get("documents", [])
    except Exception as e:
        logger.error(f"Error reading metadata: {e}")
    
    return []


# ============================================================================
# STEP 1: FILE UPLOAD HANDLING
# ============================================================================

def process_uploaded_file(file):
    """
    Process a single uploaded file and add it to the knowledge base.
    Returns status and enables/disables subsequent steps.
    """
    global _current_document, _current_chunks
    
    if file is None:
        return (
            "❌ No file selected. Please upload a document first.",
            gr.update(visible=False),  # summary section
            gr.update(visible=False),  # qa section
            gr.update(visible=False),  # chunks section
            gr.update(interactive=False),  # summary btn
            gr.update(interactive=False),  # qa section components
            "",  # clear summary
            [],  # clear chat
            ""   # clear chunks
        )
    
    if not system_ready:
        return (
            "⚠️ System not ready. Please check your OpenAI API key in the .env file.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            [],
            ""
        )
    
    try:
        ingestor = get_ingestor()
        chunker = get_chunker()
        vector_store = get_vector_store()
    except Exception as e:
        return (
            f"❌ Initialization error: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            [],
            ""
        )
    
    try:
        # Handle both string paths and file objects
        if isinstance(file, str):
            file_path = Path(file)
        elif hasattr(file, 'name'):
            file_path = Path(file.name)
        else:
            return (
                "❌ Unknown file format",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",
                [],
                ""
            )
        
        filename = file_path.name
        
        # Check extension
        extension = file_path.suffix.lower()
        if extension not in Config.SUPPORTED_EXTENSIONS:
            return (
                f"❌ Unsupported file type: {extension}\n\nSupported formats: PDF, DOCX, TXT, PNG, JPG",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",
                [],
                ""
            )
        
        # Read content
        with open(file_path, "rb") as f:
            content = f.read()
        
        if not content:
            return (
                f"❌ The file '{filename}' appears to be empty.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",
                [],
                ""
            )
        
        # Save file
        saved_path = ingestor.save_uploaded_file(filename, content)
        
        # Ingest document
        document = ingestor.ingest_file(saved_path, content)
        
        if document is None:
            return (
                f"❌ Failed to extract content from '{filename}'.\nThe file may be corrupted or password-protected.",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "",
                [],
                ""
            )
        
        is_duplicate = document.get("duplicate", False)
        
        if not is_duplicate:
            # Chunk document
            chunks = chunker.chunk_document(document)
            chunks = chunker.merge_small_chunks(chunks)
            
            if not chunks:
                return (
                    f"❌ No content could be extracted from '{filename}'.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "",
                    [],
                    ""
                )
            
            # Add to vector store
            vector_store.add_chunks(chunks)
            
            # Update metadata
            update_metadata({
                "document_id": document["document_id"],
                "filename": document["filename"],
                "hash": document["hash"],
                "modality": document["modality"],
                "upload_timestamp": document["upload_timestamp"],
                "chunk_count": len(chunks)
            })
            
            _current_document = document
            _current_chunks = chunks
        else:
            # For duplicates, load existing chunks
            doc_id = document.get("document_id") or document.get("hash", "")[:16]
            _current_chunks = vector_store.get_document_chunks(doc_id)
            _current_document = document
        
        # Determine file type icon
        modality = document.get("modality", "text")
        if modality == "image":
            type_icon = "🖼️"
            type_name = "Image"
        elif modality == "pdf":
            type_icon = "📄"
            type_name = "PDF"
        elif modality == "docx":
            type_icon = "📝"
            type_name = "Word Document"
        else:
            type_icon = "📃"
            type_name = "Text File"
        
        chunk_count = len(_current_chunks) if _current_chunks else 0
        
        if is_duplicate:
            status_message = f"""
## ✅ Document Already Indexed

| Property | Value |
|----------|-------|
| **Filename** | {type_icon} {filename} |
| **Type** | {type_name} |
| **Status** | Already in knowledge base |
| **Chunks** | {chunk_count} chunks available |

*This document was previously uploaded. You can proceed to get a summary or ask questions.*
"""
        else:
            status_message = f"""
## ✅ Document Successfully Processed!

| Property | Value |
|----------|-------|
| **Filename** | {type_icon} {filename} |
| **Type** | {type_name} |
| **Chunks Created** | {chunk_count} |
| **Status** | Ready for Q&A |

*Your document has been indexed and is ready. Proceed to the next steps!*
"""
        
        return (
            status_message,
            gr.update(visible=True),   # Show summary section
            gr.update(visible=True),   # Show QA section
            gr.update(visible=True),   # Show chunks section
            gr.update(interactive=True),  # Enable summary btn
            gr.update(interactive=True),  # Enable QA components
            "",  # Clear previous summary
            [],  # Clear previous chat
            ""   # Clear previous chunks
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {traceback.format_exc()}")
        return (
            f"❌ Error processing file: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            [],
            ""
        )


# ============================================================================
# STEP 2: DOCUMENT SUMMARY
# ============================================================================

def generate_document_summary():
    """Generate a comprehensive summary of the uploaded document."""
    global _current_document, _current_chunks
    
    if not _current_document or not _current_chunks:
        return "❌ No document has been uploaded yet. Please upload a document first."
    
    if not system_ready:
        return "⚠️ System not ready. Please check your OpenAI API key."
    
    try:
        client = Config.get_openai_client()
        
        # Combine chunk texts for summary (limit to prevent token overflow)
        combined_text = "\n\n".join([c["text"] for c in _current_chunks[:10]])
        
        # Truncate if too long
        if len(combined_text) > 15000:
            combined_text = combined_text[:15000] + "..."
        
        response = client.chat.completions.create(
            model=Config.LARGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a document summarization expert. Create a comprehensive yet concise summary of the provided document.

Structure your summary as follows:
1. **Overview**: A 2-3 sentence high-level summary
2. **Key Topics**: Main subjects or themes covered
3. **Important Points**: Bullet points of the most critical information
4. **Document Type**: What kind of document this appears to be

Be accurate and only include information present in the document."""
                },
                {
                    "role": "user",
                    "content": f"Please summarize this document:\n\n{combined_text}"
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        
        filename = _current_document.get("filename", "Unknown")
        chunk_count = len(_current_chunks)
        
        result = f"""
## 📋 Document Summary

**Source:** {filename}  
**Analyzed Sections:** {min(10, chunk_count)} of {chunk_count} chunks

---

{summary}

---

*Summary generated successfully. You can now ask specific questions about this document in Step 3.*
"""
        return result
        
    except Exception as e:
        logger.error(f"Summary generation error: {traceback.format_exc()}")
        return f"❌ Error generating summary: {str(e)}"


# ============================================================================
# STEP 3: QUESTION ANSWERING (GRADIO 6.0 COMPATIBLE)
# ============================================================================

def chat_with_document(message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Process a user question about the uploaded document.
    Uses Gradio 6.0 messages format with role/content dictionaries.
    """
    if not message or not message.strip():
        return "", history if history else []
    
    if history is None:
        history = []
    
    if not system_ready:
        response = "⚠️ System not ready. Please check your OpenAI API key in the .env file."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history
    
    try:
        vector_store = get_vector_store()
        
        # Check if documents are uploaded
        if not vector_store.is_initialized():
            response = """❌ **No documents uploaded yet!**

Please complete **Step 1** first:
1. Go to the **Upload Document** section
2. Upload your document (PDF, DOCX, TXT, or Image)
3. Wait for processing to complete
4. Then come back here to ask questions!"""
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history
        
        # Get orchestrator and process query
        orchestrator = get_orchestrator()
        result = orchestrator.query(message.strip())
        
        # Format response
        answer = result.get("answer", "Sorry, I couldn't process your question.")
        citations = result.get("citations", [])
        
        # Build formatted response
        response = f"{answer}"
        
        if citations:
            response += "\n\n---\n**📚 Sources:**\n"
            for i, citation in enumerate(citations, 1):
                source = citation.get("source", "Unknown")
                page = citation.get("page")
                if page:
                    response += f"  [{i}] {source} (Page {page})\n"
                else:
                    response += f"  [{i}] {source}\n"
        
        # Append using Gradio 6.0 messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        # Save question and answer to PostgreSQL
        db = SessionLocal()
        try:
            record = ChatHistory(
                session_id="gradio-session",
                question=message.strip(),
                answer=response
            )
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
            logger.error(f"Failed to save chat history: {traceback.format_exc()}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        error_response = f"❌ Error processing your question: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_response})
    
    return "", history


def clear_chat():
    """Clear chat history."""
    try:
        get_orchestrator().reset_conversation()
    except:
        pass
    return [], ""



# ============================================================================
# STEP 4: VIEW CHUNKS
# ============================================================================

def get_document_chunks():
    """Get formatted view of document chunks."""
    global _current_chunks
    
    if not _current_chunks:
        return "❌ No document has been uploaded yet. Please upload a document in Step 1."
    
    chunks_display = f"## 📦 Document Chunks\n\n**Total Chunks:** {len(_current_chunks)}\n\n---\n\n"
    
    for i, chunk in enumerate(_current_chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "N/A")
        chunk_id = chunk.get("chunk_id", "N/A")
        text = chunk.get("text", "")[:500]  # Limit preview length
        token_count = chunk.get("token_count", 0)
        
        if len(chunk.get("text", "")) > 500:
            text += "..."
        
        chunks_display += f"""
### Chunk {i}

| Property | Value |
|----------|-------|
| **Chunk ID** | `{chunk_id}` |
| **Source** | {source} |
| **Page** | {page} |
| **Tokens** | {token_count} |

**Content Preview:**
```
{text}
```

---

"""
    
    return chunks_display


# ============================================================================
# SYSTEM STATUS
# ============================================================================

def get_system_status() -> str:
    """Get current system status and statistics."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        documents = get_uploaded_documents()
        
        # Format document list
        doc_list = ""
        if documents:
            for doc in documents[-10:]:  # Show last 10
                modality_icon = "🖼️" if doc.get("modality") == "image" else "📄"
                doc_list += f"  • {modality_icon} {doc.get('filename', 'Unknown')} ({doc.get('chunk_count', 0)} chunks)\n"
        else:
            doc_list = "  • No documents uploaded yet\n"
        
        status = f"""
## 📊 System Status

### Knowledge Base Statistics
| Metric | Value |
|--------|-------|
| **Total Documents** | {stats['unique_documents']} |
| **Total Chunks** | {stats['total_chunks']} |
| **Cached Embeddings** | {stats['cached_embeddings']} |
| **Vector Index Size** | {stats['total_vectors']} |

### Recent Documents
{doc_list}

### Configuration
| Setting | Value |
|---------|-------|
| **Chunk Size** | ~{Config.CHUNK_SIZE} tokens |
| **Chunk Overlap** | ~{Config.CHUNK_OVERLAP} tokens |
| **Max Retrieval** | {Config.TOP_K_RETRIEVAL} chunks |
| **Conversation Window** | {Config.CONVERSATION_WINDOW} turns |
| **Answer Model** | {Config.LARGE_MODEL} |
| **Helper Model** | {Config.SMALL_MODEL} |

### Supported File Types
| Format | Description |
|--------|-------------|
| **PDF** | Page-aware text extraction |
| **DOCX** | Paragraph-aware extraction |
| **TXT** | Direct text ingestion |
| **PNG/JPG** | Vision-based OCR extraction |
"""
        return status.strip()
        
    except Exception as e:
        return f"❌ Error getting status: {str(e)}"


# ============================================================================
# GRADIO UI - CLEAN DEFAULT THEME
# ============================================================================

with gr.Blocks(
    title="Personal Knowledge Assistant",
    theme=gr.themes.Default()  # Use default Gradio theme for best visibility
) as demo:
    
    # ==================== HEADER ====================
    gr.Markdown("""
    # 🧠 Personal Knowledge Assistant
    
    Upload documents, get instant summaries, and ask intelligent questions with AI-powered answers.
    
    ---
    
    **Workflow:** Upload Document → Generate Summary → Ask Questions → View Chunks
    
    ---
    """)
    
    # ==================== STEP 1: UPLOAD ====================
    with gr.Group():
        gr.Markdown("""
        ## Step 1️⃣: Upload Your Document
        
        Start by uploading a PDF, Word document, text file, or image.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="📁 Drop your file here or click to browse",
                    file_count="single",
                    file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
                )
                
                upload_btn = gr.Button(
                    "🚀 Process Document",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                **Supported Formats:**
                - 📄 **PDF** - Multi-page documents
                - 📝 **DOCX** - Microsoft Word files
                - 📃 **TXT** - Plain text files
                - 🖼️ **PNG/JPG** - Images with text (OCR)
                """)
            
            with gr.Column(scale=1):
                upload_status = gr.Markdown(
                    value="""
## 📤 Ready to Upload

Upload a document to begin. Once processed, you'll be able to:
- ✨ Get an AI-generated summary
- 💬 Ask questions about the content
- 📦 View the document chunks

**Max file size:** 50MB
                    """
                )
    
    # ==================== STEP 2: SUMMARY ====================
    summary_section = gr.Group(visible=False)
    with summary_section:
        gr.Markdown("""
        ---
        
        ## Step 2️⃣: Document Summary
        
        Get an AI-generated overview of your document.
        """)
        
        with gr.Row():
            with gr.Column():
                summary_btn = gr.Button(
                    "✨ Generate Summary",
                    variant="primary",
                    size="lg",
                    interactive=False
                )
                
                summary_output = gr.Markdown(
                    value="*Click 'Generate Summary' to analyze your document*"
                )
    
    # ==================== STEP 3: Q&A ====================
    qa_section = gr.Group(visible=False)
    with qa_section:
        gr.Markdown("""
        ---
        
        ## Step 3️⃣: Ask Questions
        
        Get accurate, citation-backed answers from your document.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask any question about your document...",
                        show_label=False,
                        scale=5,
                        container=False
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        interactive=False
                    )
                
                clear_btn = gr.Button(
                    "🗑️ Clear Chat",
                    variant="secondary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("""
### 💡 Tips for Better Answers

- **Be specific** in your questions
- **Reference topics** you saw in the summary
- **Ask follow-ups** to dive deeper
- **Check citations** to verify sources

---

### 📝 Example Questions

- "What is the main topic?"
- "Summarize the key findings"
- "What does it say about [topic]?"
- "List the main points discussed"
- "What are the conclusions?"
                """)
    
    # ==================== STEP 4: CHUNKS ====================
    chunks_section = gr.Group(visible=False)
    with chunks_section:
        gr.Markdown("""
        ---
        
        ## Step 4️⃣: Document Chunks
        
        View how your document was split for processing.
        """)
        
        with gr.Row():
            chunks_btn = gr.Button(
                "📦 View Chunks",
                variant="primary"
            )
        
        chunks_output = gr.Markdown(
            value="*Click 'View Chunks' to see document segments*"
        )
    
    # ==================== SYSTEM STATUS (Accordion) ====================
    with gr.Accordion("📊 System Status & Settings", open=False):
        with gr.Row():
            refresh_btn = gr.Button(
                "🔄 Refresh Status",
                variant="secondary"
            )
        
        status_display = gr.Markdown(
            value="*Click 'Refresh' to load system status*"
        )
        
        refresh_btn.click(get_system_status, outputs=[status_display])
    
    # ==================== HELP SECTION (Accordion) ====================
    with gr.Accordion("❓ Help & Documentation", open=False):
        gr.Markdown("""
## How to Use This Application

### Step 1: Upload Documents
1. Click the upload area or drag & drop your file
2. Click **Process Document**
3. Wait for the processing to complete

### Step 2: Get Summary
1. Click **Generate Summary**
2. Review the AI-generated overview
3. Identify key topics for your questions

### Step 3: Ask Questions
1. Type your question in the input box
2. Press Enter or click **Send**
3. View answers with source citations

### Step 4: Explore Chunks
1. Click **View Chunks**
2. See how your document was segmented
3. Understand what information is available

---

## Technical Details

### How It Works
1. **Document Ingestion**: Files are parsed and text is extracted
2. **Chunking**: Text is split into ~400 token chunks with overlap
3. **Embedding**: Chunks are converted to vector embeddings
4. **Storage**: Embeddings are stored in a local FAISS index
5. **Retrieval**: Questions retrieve the most relevant chunks
6. **Answer Generation**: AI generates answers from retrieved context
7. **Citation Verification**: All claims are verified against sources

### Multi-Agent Architecture
- **Retriever Agent**: Finds relevant document chunks
- **Answer Generator**: Creates grounded answers
- **Citation Verifier**: Ensures claims have proper citations

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No documents uploaded" | Complete Step 1 first |
| "API key error" | Check your .env file |
| Empty answers | Try rephrasing your question |
| Slow responses | Large documents take longer to process |
        """)
    
    # ==================== FOOTER ====================
    gr.Markdown("""
    ---
    
    **Personal Knowledge Assistant** | Multi-Agent RAG System | Powered by OpenAI & LangGraph
    """)
    
    # ==================== EVENT HANDLERS ====================
    
    # Step 1: Upload
    upload_btn.click(
        process_uploaded_file,
        inputs=[file_upload],
        outputs=[
            upload_status,
            summary_section,
            qa_section,
            chunks_section,
            summary_btn,
            send_btn,
            summary_output,
            chatbot,
            chunks_output
        ]
    )
    
    # Step 2: Summary
    summary_btn.click(
        generate_document_summary,
        outputs=[summary_output]
    )
    
    # Step 3: Q&A
    send_btn.click(
        chat_with_document,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    msg_input.submit(
        chat_with_document,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg_input]
    )
    
    # Step 4: Chunks
    chunks_btn.click(
        get_document_chunks,
        outputs=[chunks_output]
    )
    
    # Load status on page load
    demo.load(get_system_status, outputs=[status_display])


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Personal Knowledge Assistant")
    print("="*60)
    print(f"🌐 Starting server on http://localhost:{Config.PORT}")
    print("📄 Upload documents and start asking questions!")
    print("="*60 + "\n")
    
    demo.launch(
        server_name=Config.HOST,
        server_port=Config.PORT,
        share=False,
        show_error=True
    )
