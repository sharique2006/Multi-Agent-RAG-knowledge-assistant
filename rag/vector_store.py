"""
Vector store module.
Handles local FAISS vector storage with embedding caching.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.config import Config
from utils.logger import logger


class VectorStore:
    """Local FAISS vector store with embedding caching."""
    
    def __init__(self):
        self.client = Config.get_openai_client()
        self.index = None
        self.chunks: List[Dict] = []
        self.embedding_cache: Dict[str, List[float]] = {}
        self._load_state()
    
    def _load_state(self):
        """Load existing index and metadata."""
        try:
            # Load FAISS index
            index_path = Config.VECTORS_DIR / "index.faiss"
            if index_path.exists():
                import faiss
                self.index = faiss.read_index(str(index_path))
                logger.info("Loaded existing FAISS index")
            
            # Load chunk metadata
            chunks_path = Config.VECTORS_DIR / "chunks.json"
            if chunks_path.exists():
                with open(chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                logger.info(f"Loaded {len(self.chunks)} chunks from metadata")
            
            # Load embedding cache
            cache_path = Config.VECTORS_DIR / "embedding_cache.json"
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
                    
        except Exception as e:
            logger.error(f"Error loading vector store state: {str(e)}")
    
    def _save_state(self):
        """Save index and metadata to disk."""
        try:
            import faiss
            
            # Save FAISS index
            if self.index is not None:
                index_path = Config.VECTORS_DIR / "index.faiss"
                faiss.write_index(self.index, str(index_path))
            
            # Save chunk metadata
            chunks_path = Config.VECTORS_DIR / "chunks.json"
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2)
            
            # Save embedding cache
            cache_path = Config.VECTORS_DIR / "embedding_cache.json"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self.embedding_cache, f)
                
            logger.info("Saved vector store state")
            
        except Exception as e:
            logger.error(f"Error saving vector store state: {str(e)}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        text_hash = self._get_text_hash(text)
        
        # Check cache
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate new embedding
        try:
            response = self.client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Batch embed uncached texts
        if texts_to_embed:
            try:
                response = self.client.embeddings.create(
                    model=Config.EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                
                for j, embedding_data in enumerate(response.data):
                    idx = indices_to_embed[j]
                    embedding = embedding_data.embedding
                    embeddings[idx] = embedding
                    
                    # Cache
                    text_hash = self._get_text_hash(texts_to_embed[j])
                    self.embedding_cache[text_hash] = embedding
                    
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {str(e)}")
                raise
        
        return embeddings
    
    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks to the vector store.
        
        Returns:
            Number of chunks added.
        """
        import faiss
        
        if not chunks:
            return 0
        
        # Filter out duplicates (by chunk_id)
        existing_ids = {c["chunk_id"] for c in self.chunks}
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        
        if not new_chunks:
            logger.info("All chunks already exist in store")
            return 0
        
        # Get embeddings
        texts = [c["text"] for c in new_chunks]
        embeddings = self._get_embeddings_batch(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Initialize or add to index
        if self.index is None:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
        else:
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
        
        # Store chunk metadata
        self.chunks.extend(new_chunks)
        
        # Save state
        self._save_state()
        
        logger.info(f"Added {len(new_chunks)} chunks to vector store")
        return len(new_chunks)
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunks with similarity scores.
        """
        import faiss
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        top_k = top_k or Config.TOP_K_RETRIEVAL
        top_k = min(top_k, self.index.ntotal)
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, top_k)
            
            # Build results with deterministic ordering
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks) and idx >= 0:
                    chunk = self.chunks[idx].copy()
                    chunk["similarity_score"] = float(score)
                    chunk["rank"] = i + 1
                    results.append(chunk)
            
            # Sort by score (deterministic)
            results.sort(key=lambda x: (-x["similarity_score"], x["chunk_id"]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document."""
        return [c for c in self.chunks if c["document_id"] == document_id]
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        Note: FAISS doesn't support deletion well, so we rebuild the index.
        
        Returns:
            Number of chunks deleted.
        """
        import faiss
        
        # Find chunks to keep
        chunks_to_keep = [c for c in self.chunks if c["document_id"] != document_id]
        deleted_count = len(self.chunks) - len(chunks_to_keep)
        
        if deleted_count == 0:
            return 0
        
        # Rebuild index
        self.chunks = chunks_to_keep
        
        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            embeddings = self._get_embeddings_batch(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
        else:
            self.index = None
        
        self._save_state()
        
        logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
        return deleted_count
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal if self.index else 0,
            "cached_embeddings": len(self.embedding_cache),
            "unique_documents": len(set(c["document_id"] for c in self.chunks))
        }
    
    def is_initialized(self) -> bool:
        """Check if vector store has data."""
        return self.index is not None and self.index.ntotal > 0
