"""
Answer Generator Agent.
Generates answers strictly from retrieved context without hallucination.
"""

from typing import Dict, List, Optional
import json

from utils.config import Config
from utils.logger import logger


class AnswerGeneratorAgent:
    """
    Agent responsible for:
    - Generating answers strictly from retrieved context
    - Never hallucinating
    - Explicitly refusing when answer not found
    """
    
    SYSTEM_PROMPT = """You are a precise question-answering assistant. Your task is to answer questions based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context to answer
2. NEVER make up or infer information not explicitly stated in the context
3. If the answer is not in the context, respond with: "The answer is not present in the uploaded documents."
4. Include citations for EVERY factual claim using the format [ChunkID: xxx]
5. Be concise but complete
6. Maintain the same language style as the question

CONTEXT FORMAT:
Each context chunk has:
- Source: The filename it came from
- Page: Page number (if applicable)
- ChunkID: Unique identifier for citation

YOUR RESPONSE FORMAT:
Provide your answer with inline citations. Example:
"According to the document, the revenue increased by 15% [ChunkID: abc123_chunk_0]. This was primarily due to new product launches [ChunkID: abc123_chunk_1]."

If the answer is not found:
"The answer is not present in the uploaded documents."
"""

    def __init__(self):
        self.client = Config.get_openai_client()
    
    def generate(
        self, 
        query: str, 
        context: str, 
        chunks: List[Dict],
        conversation_context: List[Dict] = None
    ) -> Dict:
        """
        Generate an answer from retrieved context.
        
        Args:
            query: User's question
            context: Formatted retrieval context
            chunks: Original chunk data for citation extraction
            conversation_context: Previous conversation messages
            
        Returns:
            Dict with answer and metadata
        """
        # Handle empty context
        if not context or not chunks:
            return {
                "success": True,
                "answer": "The answer is not present in the uploaded documents.",
                "raw_answer": "The answer is not present in the uploaded documents.",
                "has_answer": False,
                "chunk_ids_used": []
            }
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ]
            
            # Add conversation context if available
            if conversation_context:
                # Only add last few turns
                for msg in conversation_context[-4:]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current query with context
            user_message = f"""CONTEXT:
{context}

---

QUESTION: {query}

Please answer the question based ONLY on the context provided above. Include citations using [ChunkID: xxx] format."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate answer using stronger model
            response = self.client.chat.completions.create(
                model=Config.LARGE_MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.2
            )
            
            raw_answer = response.choices[0].message.content.strip()
            
            # Check if answer indicates no information found
            no_answer_phrases = [
                "not present in the uploaded documents",
                "not found in the context",
                "no information about",
                "context does not contain",
                "cannot find",
                "not mentioned in"
            ]
            
            has_answer = not any(
                phrase.lower() in raw_answer.lower() 
                for phrase in no_answer_phrases
            )
            
            # Extract used chunk IDs from answer
            chunk_ids_used = self._extract_citations(raw_answer, chunks)
            
            return {
                "success": True,
                "answer": raw_answer,
                "raw_answer": raw_answer,
                "has_answer": has_answer,
                "chunk_ids_used": chunk_ids_used
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            return {
                "success": False,
                "answer": "An error occurred while generating the answer.",
                "raw_answer": "",
                "has_answer": False,
                "chunk_ids_used": [],
                "error": str(e)
            }
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[str]:
        """Extract chunk IDs that were cited in the answer."""
        cited_ids = []
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            # Check various citation formats
            if chunk_id and (
                f"[ChunkID: {chunk_id}]" in answer or
                f"[{chunk_id}]" in answer or
                chunk_id in answer
            ):
                cited_ids.append(chunk_id)
        
        return cited_ids
    
    def generate_with_retry(
        self, 
        query: str, 
        context: str, 
        chunks: List[Dict],
        conversation_context: List[Dict] = None,
        max_retries: int = 1
    ) -> Dict:
        """Generate answer with retry on failure."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            result = self.generate(query, context, chunks, conversation_context)
            
            if result["success"]:
                return result
            
            last_error = result.get("error", "Unknown error")
            logger.warning(f"Generation attempt {attempt + 1} failed: {last_error}")
        
        # Return failure after all retries
        return {
            "success": False,
            "answer": "Failed to generate answer after multiple attempts.",
            "raw_answer": "",
            "has_answer": False,
            "chunk_ids_used": [],
            "error": last_error
        }
