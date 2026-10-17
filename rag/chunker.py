"""
Text chunking module.
Splits documents into overlapping chunks for embedding.
"""

from typing import Dict, List, Optional
import tiktoken

from utils.config import Config
from utils.logger import logger


class TextChunker:
    """Handles text chunking with token-aware splitting."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Split document into overlapping chunks.
        
        Args:
            document: Dict with text, filename, document_id, etc.
            
        Returns:
            List of chunk dictionaries with metadata.
        """
        text = document.get("text", "")
        if not text.strip():
            return []
        
        pages = document.get("pages", [])
        
        # If we have page information, chunk by page first
        if pages:
            return self._chunk_with_pages(document, pages)
        else:
            return self._chunk_text(document, text)
    
    def _chunk_with_pages(self, document: Dict, pages: List[Dict]) -> List[Dict]:
        """Chunk document while preserving page information."""
        all_chunks = []
        chunk_counter = 0
        
        for page_info in pages:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")
            
            if not page_text.strip():
                continue
            
            page_chunks = self._split_text(page_text)
            
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                    
                all_chunks.append({
                    "chunk_id": f"{document['document_id']}_chunk_{chunk_counter}",
                    "document_id": document["document_id"],
                    "source": document["filename"],
                    "page": page_num,
                    "modality": document.get("modality", "text"),
                    "upload_timestamp": document.get("upload_timestamp", ""),
                    "text": chunk_text.strip(),
                    "token_count": self.count_tokens(chunk_text)
                })
                chunk_counter += 1
        
        return all_chunks
    
    def _chunk_text(self, document: Dict, text: str) -> List[Dict]:
        """Chunk text without page information."""
        chunks = self._split_text(text)
        result = []
        
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            result.append({
                "chunk_id": f"{document['document_id']}_chunk_{i}",
                "document_id": document["document_id"],
                "source": document["filename"],
                "page": None,
                "modality": document.get("modality", "text"),
                "upload_timestamp": document.get("upload_timestamp", ""),
                "text": chunk_text.strip(),
                "token_count": self.count_tokens(chunk_text)
            })
        
        return result
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        Tries to preserve sentence boundaries.
        """
        if not text.strip():
            return []
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < total_tokens:
            # Calculate end position
            end = min(start + self.chunk_size, total_tokens)
            
            # Get chunk tokens and decode
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Try to find a good break point (sentence boundary)
            if end < total_tokens:
                chunk_text = self._find_break_point(chunk_text)
                # Recalculate actual token count
                actual_tokens = len(self.tokenizer.encode(chunk_text))
                end = start + actual_tokens
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move start with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= total_tokens:
                break
        
        return chunks
    
    def _find_break_point(self, text: str) -> str:
        """Find a good break point at sentence boundary."""
        # Look for sentence-ending punctuation
        for delimiter in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
            last_pos = text.rfind(delimiter)
            if last_pos > len(text) * 0.5:  # Only if in latter half
                return text[:last_pos + 1]
        
        # Look for paragraph break
        para_break = text.rfind("\n\n")
        if para_break > len(text) * 0.5:
            return text[:para_break]
        
        # Look for any newline
        newline = text.rfind("\n")
        if newline > len(text) * 0.7:
            return text[:newline]
        
        # Return as-is if no good break point
        return text
    
    def merge_small_chunks(self, chunks: List[Dict], min_tokens: int = 50) -> List[Dict]:
        """Merge chunks that are too small."""
        if not chunks:
            return []
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            if buffer is None:
                buffer = chunk.copy()
            elif buffer["token_count"] < min_tokens:
                # Merge with current chunk
                buffer["text"] = buffer["text"] + "\n\n" + chunk["text"]
                buffer["token_count"] = self.count_tokens(buffer["text"])
            else:
                merged.append(buffer)
                buffer = chunk.copy()
        
        if buffer:
            merged.append(buffer)
        
        return merged
