"""
LangGraph Node Functions

This module contains the core logic nodes for the chat assistant workflow:
- analyze_query_node: Query understanding and ambiguity detection
- summarize_node: Auto-summarization when token threshold is exceeded
- answer_node: Response generation using context
- clarify_node: Clarification request handler
"""

import os
from typing import Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from schemas import GraphState, SessionSummary, QueryAnalysis
from utils import count_tokens, format_summary_for_prompt, messages_to_text


# Configuration
GROQ_MODEL = "llama-3.1-8b-instant" 
TOKEN_THRESHOLD = 800


def get_groq_client() -> ChatGroq:
    """Initialize and return a Groq chat client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return ChatGroq(model=GROQ_MODEL, temperature=0.7)


def get_message_content(message) -> str:
    """
    Extract content from a message, whether it's a dict or LangChain object.
    
    Args:
        message: Dict with 'content' key or LangChain message object
        
    Returns:
        Message content as string
    """
    if isinstance(message, dict):
        return message.get("content", "")
    else:
        # LangChain message object
        return getattr(message, "content", "")


def analyze_query_node(state: GraphState) -> GraphState:
    """
    Analyze user query for ambiguity and determine needed context.
    
    This node uses Groq's structured output to detect if the user's intent
    is unclear and identify what information from session memory is needed.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with QueryAnalysis results
    """
    messages = state["messages"]
    summary = state["summary"]
    
    if not messages:
        # No messages to analyze
        state["analysis"] = QueryAnalysis(
            is_ambiguous=False,
            needed_context_from_memory=[]
        )
        return state
    
    # Get the latest user message
    latest_message = get_message_content(messages[-1]) if messages else ""
    
    # Get recent conversation context (last 5 messages)
    recent_context = messages[-5:] if len(messages) > 5 else messages
    context_text = "\n".join([
        f"{'user' if isinstance(msg, dict) and msg.get('role') == 'user' else getattr(msg, 'type', 'unknown')}: {get_message_content(msg)}"
        for msg in recent_context
    ])
    
    # Query analysis system prompt - optimized for structured output
    system_prompt = """You are an expert Query Understanding Agent. Your task is to process a user's message and output a structured analysis for a downstream chat assistant.

CONTEXTUAL DATA:
- Session Memory (Summary): Available fields are user_profile, key_facts, decisions, open_questions, todos
- Conversation Buffer (Last N messages): Recent conversation context

YOUR 3-STEP PIPELINE:

1. Rewrite: Identify if the user query is ambiguous (vague pronouns like 'it', 'those', or missing entities). If yes, rewrite it into a clear, standalone question.

2. Augment: Specify exactly which fields from the Session Memory are needed to answer this query. ONLY use these exact field names: user_profile, key_facts, decisions, open_questions, todos.

3. Clarify: If the query is still fundamentally unclear after rewriting, generate 1-3 polite clarifying questions.

STRICT OUTPUT FORMAT:
- You MUST return a single, valid JSON object only.
- DO NOT include markdown code blocks (```json), introductory text, or XML tags like <function>.
- Ensure all strings are JSON-safe. Handle apostrophes (e.g., "I'm") by using standard escape sequences if necessary.

SCHEMA:
{
  "original_query": "string",
  "is_ambiguous": boolean,
  "rewritten_query": "string or null",
  "needed_context_from_memory": ["list of valid field names or null"],
  "clarifying_questions": ["list of strings or null"],
  "final_augmented_context": "string (combined memory + recent history)"
}

CRITICAL RULES:
- If is_ambiguous is false, clarifying_questions MUST be null or empty
- needed_context_from_memory can ONLY contain: user_profile, key_facts, decisions, open_questions, todos
- Return ONLY the JSON object, nothing else"""

    user_prompt = f"""Recent conversation:
{context_text}

Latest query: {latest_message}

Analyze this query for ambiguity and determine what context is needed."""

    try:
        llm = get_groq_client()
        structured_llm = llm.with_structured_output(QueryAnalysis)
        
        # Retry logic for structured output - increased to 3 for maximum reliability
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Populate original_query from user input
                response.original_query = latest_message
                
                # Step 1-2: Build augmented context if needed
                if response.needed_context_from_memory:
                    memory_context = format_summary_for_prompt(
                        summary, 
                        response.needed_context_from_memory
                    )
                    response.final_augmented_context = (
                        f"{memory_context}\n\n"
                        f"Recent conversation:\n{context_text}"
                    )
                
                # JSON Logging for observability
                print("\n" + "="*50)
                print("QUERY ANALYSIS (Step 1-3)")
                print("="*50)
                print(response.model_dump_json(indent=2))
                print("="*50 + "\n")
                
                state["analysis"] = response
                return state
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    # Final attempt failed, raise error
                    raise last_error
                    
    except Exception as e:
        print(f"Error in analyze_query_node: {e}")
        # Fallback: assume query is clear
        state["analysis"] = QueryAnalysis(
            original_query=latest_message,
            is_ambiguous=False,
            needed_context_from_memory=[]
        )
    
    return state


def summarize_node(state: GraphState) -> GraphState:
    """
    Summarize conversation when token threshold is exceeded.
    
    This node merges new conversation information into the existing
    SessionSummary and resets the message list to keep only recent context.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with new summary and pruned messages
    """
    messages = state["messages"]
    current_summary = state["summary"]
    
    # Determine which messages need summarization
    last_summarized_index = current_summary.message_range_summarized.get("to", 0)
    messages_to_summarize = messages[last_summarized_index:]
    
    if not messages_to_summarize:
        return state
    
    # Convert messages to text for the prompt
    conversation_text = "\n".join([
        f"{'user' if isinstance(msg, dict) and msg.get('role') == 'user' else getattr(msg, 'type', 'unknown')}: {get_message_content(msg)}"
        for msg in messages_to_summarize
    ])
    
    # Summarization system prompt - Memory Management Agent role definition
    system_prompt = """### Role
You are a Memory Management Agent. Your task is to update a structured session summary by analyzing conversation history.

### Context
The current state contains a list of messages. You will receive a batch of messages that have exceeded the threshold and need to be archived into the long-term memory.

### Task
1. Analyze the provided message batch (excluding the 5 most recent messages which are kept for immediate context).
2. Merge the new information from these messages into the existing 'SessionSummary' object.
3. Maintain the structured fields: user_profile, key_facts, decisions, open_questions, and todos.

### Guidelines
- Be concise: Use bullet points for facts and decisions.
- Avoid Redundancy: If information in the new messages contradicts the old summary, prioritize the most recent data.
- Schema Integrity: Output MUST be a valid JSON matching the SessionSummary schema.

### Additional Rules
- user_profile: Update name and persistent preferences. Budget MUST be a STRING (e.g., "$3000", not 3000).
- key_facts: Extract new factual data (dates, locations, constraints).
- decisions: Log definitive choices made by the user.
- open_questions: Track unresolved questions or unclear user intent.
- todos: Action items the user needs to complete.
- message_range_summarized: MUST be an object with "from" and "to" integers representing message indices.

### Critical - ONLY Include Explicitly Stated Information
- Do NOT infer, assume, or extrapolate facts not directly stated
- Keep fields EMPTY if the conversation doesn't provide clear evidence
- In early-stage sessions (few messages, ambiguous queries), most fields should remain empty
- Remove items from open_questions if they've been answered
- Remove items from todos if they've been completed

### Strict Output Format
- Return ONLY a raw JSON object. No markdown blocks (```json), no tags, no preamble.
- If a field has no data, return an empty list [] or empty dict {}.
- All values in user_profile MUST be strings.

### Schema
{
  "user_profile": {"name": "string", "prefers": "string", "budget": "string"},
  "key_facts": ["string"],
  "decisions": ["string"],
  "open_questions": ["string"],
  "todos": ["string"],
  "message_range_summarized": {"from": int, "to": int}
}

### Critical Rules
- DO NOT wrap the response in extra fields like "example_name"
- Return ONLY the SessionSummary object, nothing else
- All user_profile values MUST be strings (e.g., "budget": "$3000", not "budget": 3000)"""

    user_prompt = f"""Current Summary (to be updated):
{current_summary.model_dump_json(indent=2)}

Messages to Archive (new conversation to merge):
{conversation_text}

Update the summary by merging information from the messages to archive into the current summary."""

    try:
        llm = get_groq_client()
        structured_llm = llm.with_structured_output(SessionSummary)
        
        # Retry logic for structured output - increased to 3 for maximum reliability
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                
                # Update message range
                response.message_range_summarized = {
                    "from": 0,
                    "to": len(messages)
                }
                
                state["summary"] = response
                
                # JSON Logging for observability
                print("\n" + "="*50)
                print("SESSION SUMMARY (Auto-Summarization Triggered)")
                print("="*50)
                print(response.model_dump_json(indent=2))
                print("="*50 + "\n")
                
                # Keep only last 5 messages for context
                state["messages"] = messages[-5:]
                
                # Recalculate token count
                state["current_token_count"] = count_tokens(messages_to_text(state["messages"]))
                
                return state
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    # Final attempt failed, raise error
                    raise last_error
        
    except Exception as e:
        print(f"Error in summarize_node: {e}")
        # Keep existing summary on error
        pass
    
    return state


def answer_node(state: GraphState) -> GraphState:
    """
    Generate response using relevant context from memory.
    
    This node constructs a prompt with recent messages and specific
    memory fields, then generates a contextual response.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with assistant's response added to messages
    """
    messages = state["messages"]
    summary = state["summary"]
    analysis = state["analysis"]
    
    if not messages:
        return state
    
    # Get recent conversation (last 10 messages)
    recent_messages = messages[-10:] if len(messages) > 10 else messages
    
    # Format relevant memory context
    memory_context = format_summary_for_prompt(
        summary, 
        analysis.needed_context_from_memory or []
    )
    
    # Build conversation history for prompt
    conversation_history = "\n".join([
        f"{'user' if isinstance(msg, dict) and msg.get('role') == 'user' else getattr(msg, 'type', 'unknown')}: {get_message_content(msg)}"
        for msg in recent_messages[:-1]  # Exclude the latest message
    ])
    
    latest_query = get_message_content(messages[-1])
    
    # System prompt
    system_prompt = """You are a helpful AI assistant with access to conversation history and session memory.

Use the provided context to give accurate, relevant responses. If the memory contains relevant information, incorporate it naturally into your response.

Be conversational, friendly, and helpful.

IMPORTANT GUIDELINES:
- Be PROACTIVE and ACTION-ORIENTED: Provide helpful information, suggestions, and next steps
- If the user's request is general, provide comprehensive general information first
- Offer checklists, frameworks, and structured guidance when appropriate
- If you need more details for specifics, provide general value FIRST, then ask natural follow-up questions
- Frame follow-up questions conversationally, not as an interrogation
- Default to being helpful rather than waiting for perfect information

SPECIAL CASE - If the user's query is still unclear after previous clarification attempts:
- Acknowledge the ambiguity politely
- Provide the MOST HELPFUL general answer you can based on available context
- Suggest specific ways the user can rephrase for better assistance"""

    # User prompt with context
    user_prompt_parts = []
    
    if memory_context:
        user_prompt_parts.append(memory_context)
        user_prompt_parts.append("\n---\n")
    
    if conversation_history:
        user_prompt_parts.append(f"Recent conversation:\n{conversation_history}\n\n")
    
    user_prompt_parts.append(f"User: {latest_query}")
    
    user_prompt = "".join(user_prompt_parts)
    
    try:
        llm = get_groq_client()
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # With add_messages reducer, return new message instead of append
        new_message = {"role": "assistant", "content": response.content}
        
        # Update state and return - add_messages will merge the new message
        state["current_token_count"] = count_tokens(messages_to_text(state["messages"]) + f"\nassistant: {response.content}")
        state["clarification_count"] = 0  # Reset on successful answer
        
        return {
            **state,
            "messages": [new_message]
        }
        
    except Exception as e:
        print(f"Error in answer_node: {e}")
        # Return error message
        error_msg = {"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}"}
        return {
            **state,
            "messages": [error_msg]
        }
    
    return state


def clarify_node(state: GraphState) -> GraphState:
    """
    Request clarification from the user for ambiguous queries.
    
    Increments the clarification counter to prevent infinite loops.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with clarification request
    """
    analysis = state["analysis"]
    clarification_count = state.get("clarification_count", 0)
    
    # Increment clarification counter to prevent loops
    state["clarification_count"] = clarification_count + 1
    
    if not analysis.clarifying_questions or len(analysis.clarifying_questions) == 0:
        # Fallback clarification
        clarification = "I'm not sure I understand. Could you please provide more details?"
    else:
        # Present questions conversationally, not as numbered list
        if len(analysis.clarifying_questions) == 1:
            clarification = analysis.clarifying_questions[0]
        else:
            clarification = "I need some clarification:\n\n" + "\n".join([
                f"- {q}" 
                for q in analysis.clarifying_questions
            ])
    
    # With add_messages reducer, return new message
    new_message = {"role": "assistant", "content": clarification}
    state["clarification_count"] = state.get("clarification_count", 0) + 1
    
    return {
        **state,
        "messages": [new_message]
    }
