"""
Tests for the Hugging Face client functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agent_intro.hf_client import get_hf_client, call_llm


class TestHFClient:
    """Test cases for Hugging Face client setup."""

    @patch.dict('os.environ', {'HUGGING_FACE_API_KEY': 'test_key'})
    @patch('agent_intro.hf_client.HuggingFaceEndpoint')
    @patch('agent_intro.hf_client.ChatHuggingFace')
    def test_get_hf_client_with_env_key(self, mock_chat_hf, mock_hf_endpoint):
        """Test client creation with environment variable API key."""
        mock_endpoint = Mock()
        mock_hf_endpoint.return_value = mock_endpoint
        mock_chat_model = Mock()
        mock_chat_hf.return_value = mock_chat_model

        client = get_hf_client()

        mock_hf_endpoint.assert_called_once_with(
            model="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0.7,
            max_new_tokens=512,
            huggingfacehub_api_token="test_key"
        )
        mock_chat_hf.assert_called_once_with(llm=mock_endpoint)
        assert client == mock_chat_model

    @patch('agent_intro.hf_client.HuggingFaceEndpoint')
    @patch('agent_intro.hf_client.ChatHuggingFace')
    def test_get_hf_client_with_explicit_key(self, mock_chat_hf, mock_hf_endpoint):
        """Test client creation with explicitly provided API key."""
        mock_endpoint = Mock()
        mock_hf_endpoint.return_value = mock_endpoint
        mock_chat_model = Mock()
        mock_chat_hf.return_value = mock_chat_model

        client = get_hf_client(api_key="explicit_key")

        mock_hf_endpoint.assert_called_once_with(
            model="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0.7,
            max_new_tokens=512,
            huggingfacehub_api_token="explicit_key"
        )
        assert client == mock_chat_model

    @patch.dict('os.environ', {}, clear=True)
    def test_get_hf_client_no_api_key(self):
        """Test client creation without API key raises error."""
        with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY"):
            get_hf_client()

    @patch('agent_intro.hf_client.HuggingFaceEndpoint')
    @patch('agent_intro.hf_client.ChatHuggingFace')
    def test_get_hf_client_custom_params(self, mock_chat_hf, mock_hf_endpoint):
        """Test client creation with custom parameters."""
        mock_endpoint = Mock()
        mock_hf_endpoint.return_value = mock_endpoint

        get_hf_client(
            model_name="custom/model",
            temperature=0.5,
            max_new_tokens=256,
            api_key="test_key"
        )

        mock_hf_endpoint.assert_called_once_with(
            model="custom/model",
            temperature=0.5,
            max_new_tokens=256,
            huggingfacehub_api_token="test_key"
        )


class TestCallLLM:
    """Test cases for LLM calling functionality."""

    def test_call_llm_simple_response(self):
        """Test call_llm with simple text response."""
        mock_chat_model = Mock()
        mock_response = Mock()
        mock_response.content = "Hello, how can I help you?"
        mock_response.tool_calls = None
        mock_chat_model.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = call_llm(mock_chat_model, messages)

        assert result["role"] == "assistant"
        assert result["content"] == "Hello, how can I help you?"
        assert "tool_calls" not in result

    def test_call_llm_with_tool_calls(self):
        """Test call_llm with tool calling response."""
        mock_chat_model = Mock()
        mock_response = Mock()
        mock_response.content = None

        # Mock tool calls
        mock_tool_call = {
            "id": "call_123",
            "name": "add",
            "args": {"a": 3, "b": 12}
        }
        mock_response.tool_calls = [mock_tool_call]

        mock_chat_model.bind_tools.return_value.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "What is 3 plus 12?"}]
        tools = [{"type": "function", "function": {"name": "add"}}]
        result = call_llm(mock_chat_model, messages, tools)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "add"

    def test_call_llm_error_handling(self):
        """Test call_llm error handling."""
        mock_chat_model = Mock()
        mock_chat_model.invoke.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Hello"}]
        result = call_llm(mock_chat_model, messages)

        assert result["role"] == "assistant"
        assert "Error:" in result["content"]
        assert "API Error" in result["content"]

    def test_call_llm_with_tools_no_calls(self):
        """Test call_llm with tools but no tool calls made."""
        mock_chat_model = Mock()
        mock_response = Mock()
        mock_response.content = "I can help with math, but you didn't ask for a calculation."
        mock_response.tool_calls = None

        mock_bound_model = Mock()
        mock_bound_model.invoke.return_value = mock_response
        mock_chat_model.bind_tools.return_value = mock_bound_model

        messages = [{"role": "user", "content": "What can you do?"}]
        tools = [{"type": "function", "function": {"name": "add"}}]
        result = call_llm(mock_chat_model, messages, tools)

        assert result["role"] == "assistant"
        assert "help with math" in result["content"]
        assert "tool_calls" not in result


if __name__ == "__main__":
    pytest.main([__file__])