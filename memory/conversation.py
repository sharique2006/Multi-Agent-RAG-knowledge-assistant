"""
Conversation memory management.
Implements sliding window memory with token safety.
"""

from typing import Dict, List, Optional
from collections import deque
import tiktoken

from utils.config import Config
from utils.logger import logger


class ConversationMemory:
    """
    Manages conversation history with sliding window.
    Keeps last N turns and ensures token safety.
    """
    
    def __init__(self, window_size: int = None, max_tokens: int = None):
        self.window_size = window_size or Config.CONVERSATION_WINDOW
        self.max_tokens = max_tokens or Config.MAX_CONTEXT_TOKENS
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.history: deque = deque(maxlen=self.window_size * 2)  # *2 for user+assistant pairs
        
        # Retrieved chunks for current query
        self.current_chunks: List[Dict] = []
    
    def add_user_message(self, message: str):
        """Add a user message to history."""
        self.history.append({
            "role": "user",
            "content": message
        })
    
    def add_assistant_message(self, message: str, citations: List[Dict] = None):
        """Add an assistant message to history."""
        self.history.append({
            "role": "assistant",
            "content": message,
            "citations": citations or []
        })
    
    def set_current_chunks(self, chunks: List[Dict]):
        """Set the retrieved chunks for current query."""
        self.current_chunks = chunks[:Config.TOP_K_RETRIEVAL]
    
    def get_context_messages(self) -> List[Dict]:
        """
        Get conversation context for LLM.
        Returns messages trimmed to fit token budget.
        """
        messages = []
        
        # Add history messages
        for msg in self.history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return self._trim_to_token_limit(messages)
    
    def get_retrieval_context(self) -> str:
        """
        Get formatted retrieval context string.
        Includes only top-k relevant chunks.
        """
        if not self.current_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(self.current_chunks, 1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page")
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            
            header = f"[Source: {source}"
            if page:
                header += f", Page {page}"
            header += f", ID: {chunk_id}]"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _trim_to_token_limit(self, messages: List[Dict]) -> List[Dict]:
        """
        Trim messages to fit within token limit.
        Drops oldest messages first.
        """
        total_tokens = sum(self._count_tokens(m["content"]) for m in messages)
        
        while total_tokens > self.max_tokens and len(messages) > 1:
            # Remove oldest message
            removed = messages.pop(0)
            total_tokens -= self._count_tokens(removed["content"])
            logger.info(f"Trimmed message to fit token limit. Remaining: {total_tokens} tokens")
        
        return messages
    
    def trim_chunks_to_fit(self, chunks: List[Dict], available_tokens: int) -> List[Dict]:
        """
        Trim chunks to fit within available token budget.
        Drops least relevant (lowest ranked) chunks first.
        """
        if not chunks:
            return []
        
        # Sort by relevance (similarity score or rank)
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: (-x.get("similarity_score", 0), x.get("rank", 999))
        )
        
        result = []
        used_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.get("token_count", self._count_tokens(chunk.get("text", "")))
            
            if used_tokens + chunk_tokens <= available_tokens:
                result.append(chunk)
                used_tokens += chunk_tokens
            else:
                logger.info(f"Dropping chunk {chunk.get('chunk_id')} due to token limit")
        
        return result
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()
        self.current_chunks = []
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "history_length": len(self.history),
            "current_chunks": len(self.current_chunks),
            "total_history_tokens": sum(
                self._count_tokens(m["content"]) for m in self.history
            ),
            "total_chunk_tokens": sum(
                c.get("token_count", 0) for c in self.current_chunks
            )
        }
