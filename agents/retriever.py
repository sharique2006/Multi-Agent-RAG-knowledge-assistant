"""
Retriever Agent.
Converts user queries into vector searches and retrieves relevant chunks.
"""

from typing import Dict, List, Optional
import json

from utils.config import Config
from utils.logger import logger
from rag.vector_store import VectorStore


class RetrieverAgent:
    """
    Agent responsible for:
    - Converting user queries to vector searches
    - Handling metadata filtering
    - Avoiding over-retrieval
    - Graceful handling of empty/irrelevant queries
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = Config.get_openai_client()
    
    def retrieve(self, query: str, filters: Dict = None) -> Dict:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's natural language query
            filters: Optional metadata filters
            
        Returns:
            Dict with retrieved chunks and metadata
        """
        # Handle empty or whitespace-only queries
        if not query or not query.strip():
            logger.info("Empty query received")
            return {
                "success": False,
                "chunks": [],
                "message": "Query is empty",
                "query_rewrite": None
            }
        
        # Check if vector store is initialized
        if not self.vector_store.is_initialized():
            logger.warning("Vector store not initialized")
            return {
                "success": False,
                "chunks": [],
                "message": "No documents have been uploaded yet",
                "query_rewrite": None
            }
        
        try:
            # Optionally rewrite query for better retrieval
            rewritten_query = self._rewrite_query(query)
            
            # Perform vector search
            chunks = self.vector_store.search(
                rewritten_query or query,
                top_k=Config.TOP_K_RETRIEVAL
            )
            
            # Apply metadata filters if provided
            if filters and chunks:
                chunks = self._apply_filters(chunks, filters)
            
            # Check relevance threshold
            if chunks:
                # Filter out very low similarity scores (lowered threshold for better recall)
                chunks = [c for c in chunks if c.get("similarity_score", 0) > 0.1]
            
            if not chunks:
                logger.info(f"No relevant chunks found for query: {query[:50]}...")
                return {
                    "success": True,
                    "chunks": [],
                    "message": "No relevant information found in uploaded documents",
                    "query_rewrite": rewritten_query
                }
            
            logger.info(f"Retrieved {len(chunks)} chunks for query")
            
            return {
                "success": True,
                "chunks": chunks,
                "message": f"Retrieved {len(chunks)} relevant chunks",
                "query_rewrite": rewritten_query
            }
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return {
                "success": False,
                "chunks": [],
                "message": f"Retrieval error: {str(e)}",
                "query_rewrite": None
            }
    
    def _rewrite_query(self, query: str) -> Optional[str]:
        """
        Optionally rewrite query for better retrieval.
        Uses smaller model for cost efficiency.
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.SMALL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a query optimizer for a document retrieval system.
Your task is to rewrite the user's query to improve search results.

Rules:
1. Keep the core meaning intact
2. Expand abbreviations
3. Add relevant synonyms if helpful
4. Remove filler words
5. If the query is already optimal, return it as-is

Return ONLY the rewritten query, nothing else."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # Sanity check - don't use if too different
            if len(rewritten) > len(query) * 3:
                return query
                
            return rewritten
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {str(e)}")
            return None
    
    def _apply_filters(self, chunks: List[Dict], filters: Dict) -> List[Dict]:
        """Apply metadata filters to chunks."""
        filtered = chunks
        
        # Filter by source filename
        if "source" in filters:
            filtered = [c for c in filtered if c.get("source") == filters["source"]]
        
        # Filter by document ID
        if "document_id" in filters:
            filtered = [c for c in filtered if c.get("document_id") == filters["document_id"]]
        
        # Filter by modality (text/image)
        if "modality" in filters:
            filtered = [c for c in filtered if c.get("modality") == filters["modality"]]
        
        # Filter by page
        if "page" in filters:
            filtered = [c for c in filtered if c.get("page") == filters["page"]]
        
        return filtered
    
    def get_context_for_generation(self, chunks: List[Dict]) -> str:
        """Format chunks as context for answer generation."""
        if not chunks:
            return ""
        
        context_parts = []
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            page = chunk.get("page")
            chunk_id = chunk.get("chunk_id", "unknown")
            text = chunk.get("text", "")
            
            header = f"[Source: {source}"
            if page:
                header += f", Page {page}"
            header += f", ChunkID: {chunk_id}]"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
