"""
Helper Functions for Chat Assistant

This module provides utility functions for:
- Token counting using tiktoken
- Summary formatting for LLM prompts
- Test data loading from JSONL files
"""

import json
import tiktoken
from typing import Any
from schemas import SessionSummary


# Initialize tiktoken encoder (cl100k_base is used by GPT-4 and similar models)
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Number of tokens in the text
        
    Example:
        >>> count_tokens("Hello world")
        2
    """
    if not text:
        return 0
    return len(ENCODER.encode(text))


def format_summary_for_prompt(summary: SessionSummary, fields: list[str]) -> str:
    """
    Dynamically extract and format requested fields from SessionSummary.
    
    This function creates a readable text representation of specific summary
    fields to include in the LLM prompt context.
    
    Args:
        summary: SessionSummary instance containing conversation context
        fields: List of field names to extract (e.g., ['user_profile', 'key_facts'])
        
    Returns:
        Formatted string with extracted summary information
        
    Example:
        >>> summary = SessionSummary(key_facts=["User likes coffee"])
        >>> format_summary_for_prompt(summary, ["key_facts"])
        '=== SESSION MEMORY ===\\nKey Facts:\\n- User likes coffee'
    """
    if not fields:
        return ""
    
    sections = ["=== SESSION MEMORY ==="]
    
    for field in fields:
        if not hasattr(summary, field):
            continue
            
        value = getattr(summary, field)
        
        # Format field name nicely (e.g., 'user_profile' -> 'User Profile')
        field_name = field.replace("_", " ").title()
        
        if isinstance(value, dict):
            if value:  # Only include non-empty dicts
                sections.append(f"\n{field_name}:")
                for k, v in value.items():
                    sections.append(f"  - {k}: {v}")
        elif isinstance(value, list):
            if value:  # Only include non-empty lists
                sections.append(f"\n{field_name}:")
                for item in value:
                    sections.append(f"  - {item}")
        else:
            sections.append(f"\n{field_name}: {value}")
    
    return "\n".join(sections) if len(sections) > 1 else ""


def load_test_data(file_path: str) -> list[dict[str, Any]]:
    """
    Load test conversation data from a JSONL file.
    
    Each line in the JSONL file should be a valid JSON object representing
    a test case with conversation history.
    
    Args:
        file_path: Path to the .jsonl file
        
    Returns:
        List of dictionaries, each representing a test case
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        
    Example:
        >>> test_cases = load_test_data("data/conversations.jsonl")
        >>> len(test_cases)
        3
    """
    test_cases = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                test_cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    return test_cases


def messages_to_text(messages: list) -> str:
    """
    Convert a list of messages to a single text string.
    
    Handles both dict messages and LangChain message objects.
    
    Args:
        messages: List of message dicts or LangChain message objects
        
    Returns:
        Concatenated text of all messages
    """
    from langchain_core.messages import BaseMessage
    
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
        elif isinstance(msg, BaseMessage):
            # LangChain message object
            role = msg.type if hasattr(msg, 'type') else 'unknown'
            content = msg.content if hasattr(msg, 'content') else ''
        else:
            role = 'unknown'
            content = str(msg)
        
        result.append(f"{role}: {content}")
    
    return "\n".join(result)
