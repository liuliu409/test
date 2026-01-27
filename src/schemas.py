"""
Data Models for Chat Assistant with Session Memory

This module defines all Pydantic v2 models for type-safe data handling:
- SessionSummary: Accumulated conversation context
- QueryAnalysis: Query understanding and ambiguity detection results
- GraphState: LangGraph state management
"""

from typing import Annotated, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator, model_validator
from langgraph.graph.message import add_messages


class SessionSummary(BaseModel):
    """
    Stores accumulated conversation context.
    
    This model maintains a structured summary of the conversation history,
    allowing the system to preserve context even after old messages are pruned.
    """
    user_profile: dict[str, str] = Field(
        default_factory=dict,
        description="User preferences, background info, and personal details"
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Important facts mentioned during the conversation"
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Decisions made during the conversation"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions that need answers"
    )
    todos: list[str] = Field(
        default_factory=list,
        description="Action items and tasks to be completed"
    )
    message_range_summarized: dict[str, int] = Field(
        default_factory=lambda: {"from": 0, "to": 0},
        description="Range of message indices that have been summarized"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_profile": {"name": "John", "prefers": "beach vacations"},
                "key_facts": ["Budget is $3000", "Traveling in July"],
                "decisions": ["Chose Thailand as destination"],
                "open_questions": ["Which hotel to book?"],
                "todos": ["Book flight", "Apply for visa"],
                "message_range_summarized": {"from": 0, "to": 15}
            }
        }
    }


class QueryAnalysis(BaseModel):
    """
    Results from query understanding and ambiguity detection.
    
    This model captures the full 3-step query understanding pipeline:
    Step 1: Detect ambiguity and rewrite query
    Step 2: Augment with context from memory
    Step 3: Generate clarifying questions if needed
    """
    original_query: str = Field(
        description="The user's original query before any processing"
    )
    is_ambiguous: bool = Field(
        description="Whether the user's intent is unclear or has missing context"
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Clarified version of the query (Step 1: Paraphrase/Rewrite)"
    )
    needed_context_from_memory: Optional[list[str]] = Field(
        default=None,
        description="List of summary fields needed (ONLY: user_profile, key_facts, decisions, open_questions, todos)"
    )
    final_augmented_context: Optional[str] = Field(
        default=None,
        description="Step 2: Final context combining last N messages + relevant memory fields"
    )
    clarifying_questions: Optional[list[str]] = Field(
        default=None,
        description="Step 3: 1-3 questions to ask if query is still ambiguous after augmentation"
    )
    
    @field_validator('needed_context_from_memory')
    @classmethod
    def validate_memory_fields(cls, v):
        """Ensure only valid SessionSummary fields are requested."""
        if v is None:
            return v
        
        valid_fields = {'user_profile', 'key_facts', 'decisions', 'open_questions', 'todos'}
        invalid_fields = [f for f in v if f not in valid_fields]
        
        if invalid_fields:
            # Filter out invalid fields instead of erroring
            return [f for f in v if f in valid_fields]
        
        return v
    
    @model_validator(mode='after')
    def validate_ambiguity_logic(self):
        """Ensure clarifying_questions only exist when is_ambiguous is True."""
        if not self.is_ambiguous and self.clarifying_questions:
            # If query is NOT ambiguous, clear clarifying questions
            self.clarifying_questions = None
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "original_query": "What about that hotel?",
                "is_ambiguous": True,
                "rewritten_query": "Which hotel in Bangkok should I book?",
                "needed_context_from_memory": ["key_facts", "decisions"],
                "final_augmented_context": "User mentioned budget of $3000, prefers Krabi...",
                "clarifying_questions": [
                    "Are you asking about hotels in Bangkok?",
                    "What's your preferred hotel budget?"
                ]
            }
        }
    }



class GraphState(TypedDict):
    """
    LangGraph state management.
    
    TypedDict for maintaining state across the LangGraph workflow nodes.
    All nodes receive and return this state structure.
    
    The messages field uses LangGraph's add_messages reducer for intelligent
    message merging and deduplication.
    """
    messages: Annotated[list[dict], add_messages]  # List of {"role": "user/assistant", "content": "..."}
    summary: SessionSummary
    analysis: QueryAnalysis
    current_token_count: int
    clarification_count: int  # Track consecutive clarification attempts to prevent loops

