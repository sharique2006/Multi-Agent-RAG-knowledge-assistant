"""
Agent Orchestrator using LangGraph.
Implements deterministic control flow for the multi-agent RAG pipeline.
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from enum import Enum
import operator

from langgraph.graph import StateGraph, END

from agents.retriever import RetrieverAgent
from agents.answer_generator import AnswerGeneratorAgent
from agents.citation_verifier import CitationVerifierAgent
from rag.vector_store import VectorStore
from memory.conversation import ConversationMemory
from utils.config import Config
from utils.logger import logger


class AgentState(TypedDict):
    """State passed between agents in the graph."""
    # Input
    query: str
    filters: Optional[Dict]
    
    # Retrieval
    chunks: List[Dict]
    retrieval_success: bool
    retrieval_message: str
    
    # Generation
    answer: str
    raw_answer: str
    has_answer: bool
    chunk_ids_used: List[str]
    
    # Verification
    verified: bool
    verification_score: float
    verification_issues: List[str]
    
    # Final output
    final_answer: str
    citations: List[Dict]
    
    # Control flow
    retry_count: int
    error: Optional[str]


class AgentOrchestrator:
    """
    Orchestrates the multi-agent RAG pipeline using LangGraph.
    
    Flow: Query -> Retriever -> Generator -> Verifier -> Output
    """
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        self.retriever = RetrieverAgent(self.vector_store)
        self.generator = AnswerGeneratorAgent()
        self.verifier = CitationVerifierAgent()
        self.memory = ConversationMemory()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        # Create graph with state schema
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("format_output", self._format_output_node)
        workflow.add_node("handle_no_results", self._handle_no_results_node)
        workflow.add_node("handle_revision", self._handle_revision_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "retrieve",
            self._should_generate,
            {
                "generate": "generate",
                "no_results": "handle_no_results"
            }
        )
        
        workflow.add_edge("generate", "verify")
        
        workflow.add_conditional_edges(
            "verify",
            self._should_revise,
            {
                "output": "format_output",
                "revise": "handle_revision"
            }
        )
        
        workflow.add_edge("handle_revision", "format_output")
        workflow.add_edge("handle_no_results", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieval node: get relevant chunks."""
        logger.info(f"Retrieval node processing query: {state['query'][:50]}...")
        
        result = self.retriever.retrieve(
            state["query"],
            state.get("filters")
        )
        
        # Update memory with retrieved chunks
        if result["chunks"]:
            self.memory.set_current_chunks(result["chunks"])
        
        return {
            **state,
            "chunks": result["chunks"],
            "retrieval_success": result["success"],
            "retrieval_message": result["message"]
        }
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """Generation node: create answer from context."""
        logger.info("Generation node processing...")
        
        context = self.retriever.get_context_for_generation(state["chunks"])
        conversation_context = self.memory.get_context_messages()
        
        result = self.generator.generate_with_retry(
            state["query"],
            context,
            state["chunks"],
            conversation_context,
            max_retries=1
        )
        
        return {
            **state,
            "answer": result["answer"],
            "raw_answer": result.get("raw_answer", ""),
            "has_answer": result["has_answer"],
            "chunk_ids_used": result["chunk_ids_used"]
        }
    
    def _verify_node(self, state: AgentState) -> AgentState:
        """Verification node: check citations."""
        logger.info("Verification node processing...")
        
        result = self.verifier.verify(
            state["answer"],
            state["chunks"]
        )
        
        return {
            **state,
            "verified": result["verified"],
            "verification_score": result["score"],
            "verification_issues": result.get("issues", []),
            "answer": result.get("clean_answer", state["answer"])
        }
    
    def _format_output_node(self, state: AgentState) -> AgentState:
        """Format final output with citations."""
        logger.info("Formatting final output...")
        
        formatted = self.verifier.format_final_answer(
            state["answer"],
            state["chunks"]
        )
        
        return {
            **state,
            "final_answer": formatted["answer"],
            "citations": formatted["citations"]
        }
    
    def _handle_no_results_node(self, state: AgentState) -> AgentState:
        """Handle case when no relevant documents found."""
        logger.info("Handling no results case...")
        
        return {
            **state,
            "answer": "The answer is not present in the uploaded documents.",
            "has_answer": False,
            "verified": True,
            "verification_score": 1.0,
            "verification_issues": []
        }
    
    def _handle_revision_node(self, state: AgentState) -> AgentState:
        """Handle case when answer needs revision."""
        logger.info("Handling revision case...")
        
        retry_count = state.get("retry_count", 0)
        
        if retry_count >= 1:
            # Max retries reached, return with disclaimer
            answer = state["answer"]
            if state["verification_issues"]:
                answer = f"{answer}\n\n[Note: Some claims may need additional verification]"
            
            return {
                **state,
                "answer": answer,
                "verified": True,  # Accept with disclaimer
                "retry_count": retry_count + 1
            }
        
        # Try to regenerate with more focus on citations
        context = self.retriever.get_context_for_generation(state["chunks"])
        
        enhanced_query = f"{state['query']}\n\nIMPORTANT: Ensure every claim has a citation from the context."
        
        result = self.generator.generate(
            enhanced_query,
            context,
            state["chunks"]
        )
        
        return {
            **state,
            "answer": result["answer"],
            "raw_answer": result.get("raw_answer", ""),
            "has_answer": result["has_answer"],
            "chunk_ids_used": result["chunk_ids_used"],
            "retry_count": retry_count + 1
        }
    
    def _should_generate(self, state: AgentState) -> str:
        """Decide whether to generate or handle no results."""
        if not state["chunks"]:
            return "no_results"
        return "generate"
    
    def _should_revise(self, state: AgentState) -> str:
        """Decide whether to output or revise."""
        if state["verified"] or state["verification_score"] >= 0.7:
            return "output"
        if state.get("retry_count", 0) >= 1:
            return "output"  # Don't retry forever
        return "revise"
    
    def query(self, query: str, filters: Dict = None) -> Dict:
        """
        Process a user query through the agent pipeline.
        
        Args:
            query: User's natural language question
            filters: Optional metadata filters
            
        Returns:
            Dict with answer and citations
        """
        # Validate vector store
        if not self.vector_store.is_initialized():
            return {
                "answer": "Please upload documents before asking questions.",
                "citations": [],
                "error": "No documents uploaded"
            }
        
        # Handle empty query
        if not query or not query.strip():
            return {
                "answer": "Please provide a question.",
                "citations": [],
                "error": "Empty query"
            }
        
        try:
            # Initialize state
            initial_state: AgentState = {
                "query": query.strip(),
                "filters": filters,
                "chunks": [],
                "retrieval_success": False,
                "retrieval_message": "",
                "answer": "",
                "raw_answer": "",
                "has_answer": False,
                "chunk_ids_used": [],
                "verified": False,
                "verification_score": 0.0,
                "verification_issues": [],
                "final_answer": "",
                "citations": [],
                "retry_count": 0,
                "error": None
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Update conversation memory
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(
                final_state["final_answer"],
                final_state["citations"]
            )
            
            return {
                "answer": final_state["final_answer"],
                "citations": final_state["citations"]
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "citations": [],
                "error": str(e)
            }
    
    def reset_conversation(self):
        """Reset conversation memory."""
        self.memory.clear()
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "memory": self.memory.get_stats()
        }
