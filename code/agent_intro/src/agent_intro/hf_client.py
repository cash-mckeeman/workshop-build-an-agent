"""
Hugging Face Client Setup

This module provides utility functions for setting up Hugging Face clients
for use with LangChain ChatHuggingFace integration.
"""

import os
from typing import Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def get_hf_client(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    api_key: Optional[str] = None
) -> ChatHuggingFace:
    """
    Create a ChatHuggingFace client using HuggingFace Inference Providers.

    Args:
        model_name: Name of the HuggingFace model to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_new_tokens: Maximum number of tokens to generate
        api_key: HuggingFace API key (uses env var if not provided)

    Returns:
        ChatHuggingFace client configured for the specified model
    """
    if api_key is None:
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGING_FACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY or HUGGING_FACE_API_KEY environment variable is required. "
                "Get your free API key at https://huggingface.co/settings/tokens"
            )

    # Create the HuggingFace endpoint
    llm = HuggingFaceEndpoint(
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=api_key
    )

    # Wrap in ChatHuggingFace for chat interface
    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def call_llm(chat_model, message_history, tool_list=None):
    """
    Simple wrapper for HuggingFace model calls with tool support.

    Args:
        chat_model: ChatHuggingFace instance
        message_history: List of messages in OpenAI format
        tool_list: Optional list of tools for function calling

    Returns:
        Dictionary with role, content, and optional tool_calls
    """
    try:
        # Bind tools if provided
        if tool_list:
            model_with_tools = chat_model.bind_tools(tool_list)
            response = model_with_tools.invoke(message_history)
        else:
            response = chat_model.invoke(message_history)

        # Format response in OpenAI-compatible format
        result = {"role": "assistant", "content": response.content}

        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            result["tool_calls"] = []
            for tool_call in response.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.get("id", f"call_{len(result['tool_calls'])}"),
                    "function": {
                        "name": tool_call["name"],
                        "arguments": str(tool_call["args"]) if isinstance(tool_call["args"], dict) else tool_call["args"]
                    },
                    "type": "function"
                })
            result["content"] = None

        return result

    except Exception as e:
        import traceback
        print(f"Error calling model: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return {"role": "assistant", "content": f"Error: {str(e)}"}