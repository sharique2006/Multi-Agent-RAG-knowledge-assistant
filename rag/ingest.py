"""
Document ingestion module.
Handles extraction of text from various file formats.
"""

import hashlib
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

from pypdf import PdfReader
from docx import Document
from PIL import Image
import io

from utils.config import Config
from utils.logger import logger


class DocumentIngestor:
    """Handles document ingestion from various file formats."""
    
    def __init__(self):
        self.client = Config.get_openai_client()
        self._processed_hashes: set = self._load_processed_hashes()
    
    def _load_processed_hashes(self) -> set:
        """Load previously processed file hashes."""
        if Config.METADATA_FILE.exists():
            try:
                with open(Config.METADATA_FILE, "r") as f:
                    metadata = json.load(f)
                    return set(doc.get("hash", "") for doc in metadata.get("documents", []))
            except (json.JSONDecodeError, KeyError):
                pass
        return set()
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()
    
    def _is_duplicate(self, file_hash: str) -> bool:
        """Check if document has already been processed."""
        return file_hash in self._processed_hashes
    
    def ingest_file(self, file_path: Path, file_content: bytes = None) -> Optional[Dict]:
        """
        Ingest a file and extract text content.
        
        Returns:
            Dict with extracted content and metadata, or None if duplicate/error.
        """
        try:
            # Read file content if not provided
            if file_content is None:
                with open(file_path, "rb") as f:
                    file_content = f.read()
            
            # Check for empty file
            if not file_content:
                logger.warning(f"Empty file: {file_path.name}")
                return None
            
            # Compute hash for deduplication
            file_hash = self._compute_hash(file_content)
            
            # Check for duplicate
            if self._is_duplicate(file_hash):
                logger.info(f"Duplicate file skipped: {file_path.name}")
                return {"duplicate": True, "filename": file_path.name}
            
            # Determine file type
            extension = file_path.suffix.lower()
            if extension not in Config.SUPPORTED_EXTENSIONS:
                logger.warning(f"Unsupported file type: {extension}")
                return None
            
            file_type = Config.SUPPORTED_EXTENSIONS[extension]
            
            # Extract text based on file type
            extracted = self._extract_content(file_path, file_content, file_type)
            
            if not extracted or not extracted.get("text"):
                logger.warning(f"No content extracted from: {file_path.name}")
                return None
            
            # Mark as processed
            self._processed_hashes.add(file_hash)
            
            # Return extracted content with metadata
            return {
                "document_id": file_hash[:16],
                "filename": file_path.name,
                "hash": file_hash,
                "modality": file_type,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "text": extracted["text"],
                "pages": extracted.get("pages", []),
                "duplicate": False
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path.name}: {str(e)}")
            return None
    
    def _extract_content(self, file_path: Path, content: bytes, file_type: str) -> Optional[Dict]:
        """Extract content based on file type."""
        extractors = {
            "pdf": self._extract_pdf,
            "docx": self._extract_docx,
            "text": self._extract_text,
            "image": self._extract_image
        }
        
        extractor = extractors.get(file_type)
        if extractor:
            return extractor(file_path, content)
        return None
    
    def _extract_pdf(self, file_path: Path, content: bytes) -> Dict:
        """Extract text from PDF with page awareness."""
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            pages = []
            full_text = []
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append({
                        "page": page_num,
                        "text": page_text.strip()
                    })
                    full_text.append(f"[Page {page_num}]\n{page_text.strip()}")
            
            return {
                "text": "\n\n".join(full_text),
                "pages": pages
            }
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return {"text": "", "pages": []}
    
    def _extract_docx(self, file_path: Path, content: bytes) -> Dict:
        """Extract text from DOCX with paragraph awareness."""
        try:
            doc = Document(io.BytesIO(content))
            paragraphs = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text.strip())
            
            return {
                "text": "\n\n".join(paragraphs),
                "pages": [{"page": 1, "text": "\n\n".join(paragraphs)}]
            }
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            return {"text": "", "pages": []}
    
    def _extract_text(self, file_path: Path, content: bytes) -> Dict:
        """Extract text from TXT file."""
        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    text = content.decode(encoding)
                    return {
                        "text": text.strip(),
                        "pages": [{"page": 1, "text": text.strip()}]
                    }
                except UnicodeDecodeError:
                    continue
            
            return {"text": "", "pages": []}
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
            return {"text": "", "pages": []}
    
    def _extract_image(self, file_path: Path, content: bytes) -> Dict:
        """Extract text from image using OpenAI Vision API."""
        try:
            # Validate image
            img = Image.open(io.BytesIO(content))
            img.verify()
            
            # Re-open after verify
            img = Image.open(io.BytesIO(content))
            
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Resize if too large (max 2000px on longest side)
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Use OpenAI Vision API
            response = self.client.chat.completions.create(
                model=Config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text content from this image. Return only the extracted text, nothing else. If there is no text, return 'NO_TEXT_FOUND'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            if extracted_text == "NO_TEXT_FOUND":
                return {"text": "", "pages": []}
            
            return {
                "text": extracted_text,
                "pages": [{"page": 1, "text": extracted_text}]
            }
            
        except Exception as e:
            logger.error(f"Image extraction error: {str(e)}")
            return {"text": "", "pages": []}
    
    def save_uploaded_file(self, filename: str, content: bytes) -> Path:
        """Save uploaded file to storage."""
        file_path = Config.UPLOADS_DIR / filename
        
        # Handle duplicate filenames
        if file_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while file_path.exists():
                file_path = Config.UPLOADS_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return file_path
