"""
Streamlit Chat Interface for Chat Assistant with Session Memory

This module provides a production-ready UI with:
- Standard chat interface
- Debug sidebar showing session state
- Test data loading capability
"""

import os
import json
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from graph import graph, initialize_state
from schemas import SessionSummary, QueryAnalysis
from utils import load_test_data, count_tokens, messages_to_text

# Load environment variables
load_dotenv()

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "conversations.jsonl"

# Page configuration
st.set_page_config(
    page_title="Chat Assistant with Session Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "default-session"

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


def convert_message_to_dict(message):
    """Convert a message to dict format, whether it's already a dict or a LangChain message object."""
    if isinstance(message, dict):
        return message
    else:
        # LangChain message object (HumanMessage, AIMessage, etc.)
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        else:
            # Fallback for unknown message types
            return {"role": "unknown", "content": str(message)}


def get_current_state():
    """Retrieve the current state from the graph."""
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Get the current state from checkpointer
        state_snapshot = graph.get_state(config)
        
        if state_snapshot and state_snapshot.values:
            return state_snapshot.values
        else:
            return initialize_state()
    except Exception as e:
        st.error(f"Error retrieving state: {e}")
        return initialize_state()


def invoke_graph(user_message: str):
    """Invoke the graph with a user message."""
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Check if we have existing state in checkpointer
        state_snapshot = graph.get_state(config)
        
        if state_snapshot and state_snapshot.values and state_snapshot.values.get("messages"):
            # State exists - only pass new message, add_messages will merge
            input_state = {
                "messages": [{"role": "user", "content": user_message}]
            }
        else:
            # First message - provide full initial state
            from graph import initialize_state
            input_state = initialize_state()
            input_state["messages"] = [{"role": "user", "content": user_message}]
        
        # Invoke the graph
        result = graph.invoke(input_state, config)
        
        return result
    except ValueError as e:
        # Configuration errors (missing API key, etc.)
        if "GROQ_API_KEY" in str(e):
            st.error("🔑 **API Key Error**: Please configure your GROQ_API_KEY in the .env file")
        else:
            st.error(f"⚙️ **Configuration Error**: {e}")
        return None
    except Exception as e:
        error_str = str(e).lower()
        
        # Rate limit errors
        if "rate_limit" in error_str or "429" in error_str:
            st.warning("⏳ **Rate Limit Reached**: Please wait a moment before sending another message. Consider upgrading your Groq plan for higher limits.")
        # Authentication errors
        elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            st.error("🔐 **Authentication Error**: Your API key appears to be invalid. Please check your GROQ_API_KEY in the .env file.")
        # Network errors
        elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
            st.error("🌐 **Connection Error**: Unable to reach the Groq API. Please check your internet connection.")
        # Generic errors
        else:
            st.error(f"⚠️ **Error**: {e}")
            st.info("If this persists, please check the console logs for more details.")
        return None


def load_test_conversation(case_index: int):
    """Load a test conversation from the JSONL file."""
    try:
        test_cases = load_test_data(str(TEST_DATA_PATH))
        
        if 0 <= case_index < len(test_cases):
            test_case = test_cases[case_index]
            
            # Reset session
            st.session_state.thread_id = f"test-session-{case_index}"
            st.session_state.display_messages = []
            
            # Load messages from test case
            for msg in test_case.get("messages", []):
                if msg["role"] == "user":
                    result = invoke_graph(msg["content"])
                    if result:
                        # Update display messages (convert to dict format)
                        st.session_state.display_messages = [
                            convert_message_to_dict(m) for m in result["messages"]
                        ]
            
            st.success(f"Loaded test case: {test_case.get('name', 'Unknown')}")
            st.rerun()
        else:
            st.error("Invalid test case index")
    except FileNotFoundError:
        st.error("Test data file not found. Please ensure data/conversations.jsonl exists.")
    except Exception as e:
        st.error(f"Error loading test data: {e}")


# Main UI
st.title("🤖 Chat Assistant with Session Memory")
st.caption("Built with LangGraph, Groq, and Streamlit")

# Sidebar for debugging and controls
with st.sidebar:
    st.header("🔧 Debug Panel")
    
    # API Key status with proactive validation
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ GROQ_API_KEY configured")
    else:
        st.error("❌ GROQ_API_KEY not found")
        st.info("Please create a .env file with your GROQ_API_KEY")
        st.warning("⚠️ Chat is disabled until API key is configured")
        # Prevent app operation without API key
        st.stop()
    
    st.divider()
    
    # Session info
    st.subheader("Session Info")
    st.text(f"Thread ID: {st.session_state.thread_id}")
    
    # Current state
    current_state = get_current_state()
    
    # Token count
    token_count = current_state.get("current_token_count", 0)
    st.metric("Current Token Count", token_count)
    
    # Progress bar for token threshold
    from nodes import TOKEN_THRESHOLD
    progress = min(token_count / TOKEN_THRESHOLD, 1.0)
    st.progress(progress, text=f"Threshold: {TOKEN_THRESHOLD}")
    
    st.divider()
    
    # Session Summary
    st.subheader("📋 Session Summary")
    summary = current_state.get("summary")
    if summary:
        with st.expander("View Summary JSON", expanded=False):
            st.json(summary.model_dump() if hasattr(summary, 'model_dump') else summary)
        
        # Display summary fields if they have content
        if hasattr(summary, 'user_profile') and summary.user_profile:
            st.write("**User Profile:**")
            for key, value in summary.user_profile.items():
                st.write(f"- {key}: {value}")
        
        if hasattr(summary, 'key_facts') and summary.key_facts:
            st.write("**Key Facts:**")
            for fact in summary.key_facts:
                st.write(f"- {fact}")
        
        if hasattr(summary, 'decisions') and summary.decisions:
            st.write("**Decisions:**")
            for decision in summary.decisions:
                st.write(f"- {decision}")
        
        if hasattr(summary, 'todos') and summary.todos:
            st.write("**To-Dos:**")
            for todo in summary.todos:
                st.write(f"- {todo}")
    
    st.divider()
    
    # Test data controls
    st.subheader("🧪 Test Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Case 1\n(Long)", use_container_width=True):
            load_test_conversation(0)
    
    with col2:
        if st.button("Case 2\n(Ambiguous)", use_container_width=True):
            load_test_conversation(1)
    
    with col3:
        if st.button("Case 3\n(Clear)", use_container_width=True):
            load_test_conversation(2)
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
        st.session_state.thread_id = f"session-{hash(str(st.session_state.get('reset_count', 0)))}"
        st.session_state.reset_count = st.session_state.get('reset_count', 0) + 1
        st.session_state.display_messages = []
        st.rerun()

# Main chat interface
st.divider()

# Display chat messages
if not st.session_state.display_messages:
    # Initialize with current state messages
    current_state = get_current_state()
    st.session_state.display_messages = [
        convert_message_to_dict(m) for m in current_state.get("messages", [])
    ]

for message in st.session_state.display_messages:
    message_dict = convert_message_to_dict(message)
    role = message_dict["role"]
    content = message_dict["content"]
    
    # Custom avatars: white for user, blue for assistant
    if role == "user":
        avatar = "👤"  # User silhouette
    else:
        avatar = "🤖"  # Robot/AI icon (will appear blue with theme)
    
    with st.chat_message(role, avatar=avatar):
        st.write(content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Display user message immediately with avatar
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    
    # Add to display messages
    st.session_state.display_messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Invoke graph
    with st.spinner("Thinking..."):
        result = invoke_graph(prompt)
    
    if result:
        # Update display messages with the latest state (convert to dict format)
        st.session_state.display_messages = [
            convert_message_to_dict(m) for m in result["messages"]
        ]
    
    st.rerun()

# Footer
st.divider()
st.caption("💡 Tip: Use the debug panel to monitor token count and session summary. Load test cases to see auto-summarization and ambiguity detection in action.")
