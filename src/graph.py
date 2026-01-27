"""
LangGraph Workflow Definition

This module defines the StateGraph flow with conditional routing:
- Analyzes queries for ambiguity
- Routes to clarification or answer generation
- Triggers auto-summarization when token threshold is exceeded
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schemas import GraphState, SessionSummary, QueryAnalysis
from nodes import (
    analyze_query_node,
    summarize_node,
    answer_node,
    clarify_node,
    TOKEN_THRESHOLD
)

# Configuration: Maximum clarification attempts before forcing best-effort answer
MAX_CLARIFICATION_ATTEMPTS = 1  # Conservative: ask once, then answer


def should_clarify(state: GraphState) -> Literal["clarify", "answer"]:
    """
    Conditional edge with guard condition to prevent infinite clarification loops.
    
    Routing logic:
    1. If query is NOT ambiguous → answer
    2. If query is ambiguous AND clarification_count < MAX → clarify
    3. If query is ambiguous AND clarification_count >= MAX → answer (force best effort)
    
    Args:
        state: Current graph state
        
    Returns:
        "clarify" if query is ambiguous and under attempt limit, "answer" otherwise
    """
    analysis = state.get("analysis")
    clarification_count = state.get("clarification_count", 0)
    
    # Guard condition: Prevent infinite loop by forcing answer after max attempts
    if clarification_count >= MAX_CLARIFICATION_ATTEMPTS:
        return "answer"  # Force best-effort answer
    
    # Normal routing: clarify if ambiguous, answer if clear
    if analysis and analysis.is_ambiguous:
        return "clarify"
    return "answer"


def should_summarize(state: GraphState) -> Literal["summarize", "end"]:
    """
    Conditional edge function to determine if summarization is needed.
    
    Args:
        state: Current graph state
        
    Returns:
        "summarize" if token count exceeds threshold, "end" otherwise
    """
    current_count = state.get("current_token_count", 0)
    if current_count > TOKEN_THRESHOLD:
        return "summarize"
    return "end"


def create_graph() -> StateGraph:
    """
    Create and compile the LangGraph workflow.
    
    Graph Flow:
    1. START -> analyze_query_node
    2. Conditional: is_ambiguous?
       - Yes -> clarify_node -> END
       - No -> answer_node
    3. After answer_node, check token threshold:
       - Exceeded -> summarize_node -> END
       - Not exceeded -> END
    
    Returns:
        Compiled StateGraph with MemorySaver checkpointer
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("summarize", summarize_node)
    
    # Define edges
    # Start with query analysis
    workflow.set_entry_point("analyze_query")
    
    # Conditional edge after analysis
    workflow.add_conditional_edges(
        "analyze_query",
        should_clarify,
        {
            "clarify": "clarify",
            "answer": "answer"
        }
    )
    
    # After clarification, end the conversation
    workflow.add_edge("clarify", END)
    
    # After answer, check if summarization is needed
    workflow.add_conditional_edges(
        "answer",
        should_summarize,
        {
            "summarize": "summarize",
            "end": END
        }
    )
    
    # After summarization, end
    workflow.add_edge("summarize", END)
    
    # Compile with memory checkpointer for persistence
    checkpointer = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    return compiled_graph


def initialize_state() -> GraphState:
    """
    Initialize a new graph state with default values.
    
    Returns:
        GraphState with empty messages and default summary
    """
    return GraphState(
        messages=[],
        summary=SessionSummary(),
        analysis=QueryAnalysis(original_query="", is_ambiguous=False),
        current_token_count=0,
        clarification_count=0  # Initialize clarification counter
    )


# Create the graph instance (singleton)
graph = create_graph()
