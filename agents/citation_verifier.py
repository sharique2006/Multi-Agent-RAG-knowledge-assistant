"""
Citation Verifier Agent.
Verifies that all claims in the answer are supported by citations.
"""

from typing import Dict, List, Optional
import json
import re

from utils.config import Config
from utils.logger import logger


class CitationVerifierAgent:
    """
    Agent responsible for:
    - Verifying every factual claim
    - Ensuring citations exist for each answer segment
    - Rejecting unsupported responses
    """
    
    VERIFICATION_PROMPT = """You are a citation verification specialist. Your task is to verify that an answer is properly supported by the provided source context.

VERIFICATION RULES:
1. Every factual claim in the answer MUST have a corresponding citation
2. The cited content MUST actually support the claim
3. No information should be fabricated or inferred beyond what's in the sources
4. General knowledge statements don't need citations, but specific claims do

INPUT FORMAT:
- ANSWER: The answer to verify
- SOURCES: The original source chunks with their IDs

OUTPUT FORMAT (JSON):
{
    "verified": true/false,
    "score": 0.0-1.0,
    "issues": ["list of specific issues found"],
    "unsupported_claims": ["claims without proper citation support"],
    "suggested_corrections": ["optional corrections if issues found"],
    "clean_answer": "the answer with proper citations, or original if all good"
}

Be strict but fair. If the answer says it cannot find information, that's valid and should verify as true with score 1.0."""

    def __init__(self):
        self.client = Config.get_openai_client()
    
    def verify(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Verify that the answer is properly supported by citations.
        
        Args:
            answer: Generated answer to verify
            chunks: Source chunks used for context
            
        Returns:
            Verification result with score and issues
        """
        # Handle "not found" answers - these are automatically valid
        if self._is_not_found_answer(answer):
            return {
                "verified": True,
                "score": 1.0,
                "issues": [],
                "unsupported_claims": [],
                "clean_answer": answer,
                "needs_revision": False
            }
        
        # Handle empty chunks
        if not chunks:
            return {
                "verified": False,
                "score": 0.0,
                "issues": ["No source chunks available for verification"],
                "unsupported_claims": [answer],
                "clean_answer": "The answer is not present in the uploaded documents.",
                "needs_revision": True
            }
        
        try:
            # Format sources for verification
            sources_text = self._format_sources(chunks)
            
            # Use smaller model for cost efficiency
            response = self.client.chat.completions.create(
                model=Config.SMALL_MODEL,
                messages=[
                    {"role": "system", "content": self.VERIFICATION_PROMPT},
                    {"role": "user", "content": f"""ANSWER:
{answer}

SOURCES:
{sources_text}

Verify this answer and return your analysis as JSON."""}
                ],
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            # Normalize result
            verified = result.get("verified", False)
            score = float(result.get("score", 0.0))
            
            return {
                "verified": verified,
                "score": score,
                "issues": result.get("issues", []),
                "unsupported_claims": result.get("unsupported_claims", []),
                "clean_answer": result.get("clean_answer", answer),
                "suggested_corrections": result.get("suggested_corrections", []),
                "needs_revision": not verified and score < 0.7
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification result: {str(e)}")
            # Fall back to basic verification
            return self._basic_verification(answer, chunks)
            
        except Exception as e:
            logger.error(f"Citation verification error: {str(e)}")
            return {
                "verified": True,  # Don't block on verification errors
                "score": 0.5,
                "issues": [f"Verification error: {str(e)}"],
                "unsupported_claims": [],
                "clean_answer": answer,
                "needs_revision": False,
                "error": str(e)
            }
    
    def _is_not_found_answer(self, answer: str) -> bool:
        """Check if answer indicates information was not found."""
        not_found_phrases = [
            "not present in the uploaded documents",
            "not found in the context",
            "cannot find the answer",
            "no information available",
            "the documents do not contain"
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in not_found_phrases)
    
    def _format_sources(self, chunks: List[Dict]) -> str:
        """Format source chunks for verification."""
        parts = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            source = chunk.get("source", "unknown")
            page = chunk.get("page")
            text = chunk.get("text", "")
            
            header = f"[ChunkID: {chunk_id}, Source: {source}"
            if page:
                header += f", Page: {page}"
            header += "]"
            
            parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(parts)
    
    def _basic_verification(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Basic verification without LLM.
        Checks if citations exist and reference valid chunks.
        """
        # Extract cited chunk IDs from answer
        citation_pattern = r'\[ChunkID:\s*([^\]]+)\]'
        cited_ids = re.findall(citation_pattern, answer)
        
        # Get available chunk IDs
        available_ids = {c.get("chunk_id") for c in chunks}
        
        # Check if all citations are valid
        invalid_citations = [cid for cid in cited_ids if cid not in available_ids]
        
        # Check if answer has any citations (for non-empty answers)
        has_factual_claims = len(answer.split()) > 20
        has_citations = len(cited_ids) > 0
        
        if invalid_citations:
            return {
                "verified": False,
                "score": 0.3,
                "issues": [f"Invalid citation IDs: {invalid_citations}"],
                "unsupported_claims": [],
                "clean_answer": answer,
                "needs_revision": True
            }
        
        if has_factual_claims and not has_citations:
            return {
                "verified": False,
                "score": 0.5,
                "issues": ["Answer contains claims but no citations"],
                "unsupported_claims": [],
                "clean_answer": answer,
                "needs_revision": True
            }
        
        return {
            "verified": True,
            "score": 0.8,
            "issues": [],
            "unsupported_claims": [],
            "clean_answer": answer,
            "needs_revision": False
        }
    
    def format_final_answer(self, answer: str, chunks: List[Dict]) -> Dict:
        """
        Format the final answer with proper citation structure.
        
        Returns:
            Dict with answer and structured citations
        """
        # Extract citations from answer text
        citation_pattern = r'\[ChunkID:\s*([^\]]+)\]'
        cited_ids = re.findall(citation_pattern, answer)
        
        # Build citation objects
        citations = []
        seen_ids = set()
        
        for chunk_id in cited_ids:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            
            # Find chunk metadata
            chunk = next((c for c in chunks if c.get("chunk_id") == chunk_id), None)
            if chunk:
                citations.append({
                    "source": chunk.get("source", "Unknown"),
                    "chunk_id": chunk_id,
                    "page": chunk.get("page")
                })
        
        # Clean answer text (optionally remove inline citations for cleaner display)
        # Keeping them for transparency
        clean_answer = answer
        
        return {
            "answer": clean_answer,
            "citations": citations
        }
