"""
Model setup page for configuring AI providers.
"""

import streamlit as st

from src.streamlit_app.components.code_display import (
    display_agent_architecture,
    display_interactive_code,
)
from src.streamlit_app.utils.formatters import format_provider_status
from src.streamlit_app.utils.session_state import mark_progress


def show_model_setup_page():
    """Display the model setup page content."""
    from src.factory import get_recommended_setup
    from src.models import check_all_providers, get_available_providers

    st.header("⚙️ Model Setup")

    st.markdown("""
    The **Model** is the "brain" of your agent - the Large Language Model (LLM) that understands
    natural language and generates responses. Let's set up your AI provider!
    """)

    # Show agent architecture with model highlighted
    display_agent_architecture()

    st.markdown("---")

    # Provider options
    st.markdown("## 🎯 Choose Your Provider")

    # Get current setup status
    setup = get_recommended_setup()
    get_available_providers()

    if setup["status"] == "ready":
        st.success(
            f"✅ **Ready!** Using {setup['recommended_provider']} with model: {setup['recommended_model']}"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(
                "Your model is configured and ready to use. You can proceed to the next page or test different providers below."
            )
        with col2:
            if st.button(
                "✅ Continue to Tools", key="continue_to_tools", type="primary"
            ):
                mark_progress("model_setup", True)
                st.success(
                    "Great! Head to the **Tools** page to learn about agent capabilities."
                )

    else:
        st.warning(
            "⚠️ No providers available. Please configure at least one provider below."
        )

    # Provider configuration sections
    st.markdown("---")
    st.markdown("## 🛠️ Provider Configuration")

    # Cloud providers
    with st.expander(
        "☁️ Cloud Providers (Recommended)", expanded=setup["status"] != "ready"
    ):
        st.markdown("""
        **Pros:** Fast, reliable, no local setup required
        **Cons:** Requires API keys, usage costs

        Choose one of these options:
        """)

        cloud_options = [
            {
                "name": "OpenAI",
                "env_var": "OPENAI_API_KEY",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                "description": "Most capable and reliable models",
                "signup": "https://platform.openai.com/",
            },
            {
                "name": "Anthropic",
                "env_var": "ANTHROPIC_API_KEY",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "description": "Excellent for reasoning and analysis",
                "signup": "https://console.anthropic.com/",
            },
            {
                "name": "HuggingFace",
                "env_var": "HUGGINGFACE_API_KEY",
                "models": [
                    "microsoft/DialoGPT-medium",
                    "meta-llama/Llama-2-7b-chat-hf",
                ],
                "description": "Open source models, free tier available",
                "signup": "https://huggingface.co/settings/tokens",
            },
        ]

        for provider in cloud_options:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"""
                **{provider["name"]}**
                - {provider["description"]}
                - Models: {", ".join(provider["models"][:2])}
                - Environment variable: `{provider["env_var"]}`
                """)

            with col2:
                if st.button(
                    f"Setup {provider['name']}", key=f"setup_{provider['name'].lower()}"
                ):
                    st.info(f"""
                    **Setup Instructions for {provider["name"]}:**

                    1. Visit {provider["signup"]}
                    2. Create an account and get your API key
                    3. Set environment variable: `export {provider["env_var"]}=your_key_here`
                    4. Restart this application

                    **Quick setup (terminal):**
                    ```bash
                    export {provider["env_var"]}=your_key_here
                    ```
                    """)

    # Local providers
    with st.expander("🏠 Local Providers (Privacy Focused)", expanded=False):
        st.markdown("""
        **Pros:** Complete privacy, no API costs, offline capable
        **Cons:** Requires installation, uses local compute resources

        Choose one of these options:
        """)

        local_options = [
            {
                "name": "Ollama",
                "description": "Easiest local setup, great performance",
                "install": "curl -fsSL https://ollama.ai/install.sh | sh",
                "models": ["llama3.2", "mistral", "codellama"],
            },
            {
                "name": "vLLM",
                "description": "High-performance inference server",
                "install": "pip install vllm",
                "models": ["Custom local models"],
            },
        ]

        for provider in local_options:
            st.markdown(f"""
            **{provider["name"]}**
            - {provider["description"]}
            - Install: `{provider["install"]}`
            - Models: {", ".join(provider["models"])}
            """)

    st.markdown("---")

    # Provider health check
    st.markdown("## 🔍 Provider Health Check")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🩺 Check All Providers", key="health_check"):
            with st.spinner("Checking provider health..."):
                health_results = check_all_providers()

                with col2:
                    st.markdown("**Health Status:**")
                    for name, result in health_results.items():
                        status_msg = format_provider_status(
                            name, result.status.value, result.message
                        )
                        if result.status.value == "healthy":
                            st.success(status_msg)
                        else:
                            st.error(status_msg)

    # Model explanation
    st.markdown("---")
    st.markdown("## 🧠 Understanding the Model Component")

    st.markdown("""
    The **Model** is the foundation of your agent. It's responsible for:

    1. **Understanding Input**: Processing and comprehending user requests
    2. **Reasoning**: Analyzing problems and determining solutions
    3. **Decision Making**: Choosing which tools to use (if any)
    4. **Response Generation**: Creating natural language responses

    Think of it as the agent's "brain" - everything else builds on top of this foundation.
    """)

    # Code example
    st.markdown("### 🔧 Code Example")

    basic_agent_code = """
from pydantic_ai import Agent

# Create a basic agent with a model
agent = Agent(
    "openai:gpt-4o-mini",  # or "anthropic:claude-3-haiku" etc.
    instructions="You are a helpful AI assistant."
)

# Use the agent
result = agent.run_sync("Hello, how are you?")
print(result.output)  # Agent responds using the model
"""

    display_interactive_code(
        basic_agent_code,
        "This is how you create a basic agent with just a model:",
        allow_edit=False,
    )

    # Model comparison
    st.markdown("### ⚖️ Model Comparison")

    comparison_data = {
        "OpenAI GPT-4": {
            "Speed": "Fast",
            "Quality": "Excellent",
            "Cost": "$$",
            "Best For": "General purpose",
        },
        "Anthropic Claude": {
            "Speed": "Fast",
            "Quality": "Excellent",
            "Cost": "$$",
            "Best For": "Analysis, reasoning",
        },
        "HuggingFace Models": {
            "Speed": "Medium",
            "Quality": "Good",
            "Cost": "$",
            "Best For": "Open source, customizable",
        },
        "Ollama (Local)": {
            "Speed": "Medium",
            "Quality": "Good",
            "Cost": "Free",
            "Best For": "Privacy, offline use",
        },
    }

    st.table(comparison_data)

    # Next steps
    st.markdown("---")
    st.markdown("## 🚀 Next Steps")

    if setup["status"] == "ready":
        st.success(
            "✅ Your model is ready! You can now create agents that understand and respond to natural language."
        )

        st.markdown("""
        **What you've accomplished:**
        - ✅ Configured an AI model provider
        - ✅ Tested the connection
        - ✅ Ready to build agents

        **Next up:** Learn how to give your agent **Tools** so it can take actions beyond just chatting!
        """)

        if st.button(
            "🛠️ Continue to Tools Page", key="continue_tools_bottom", type="primary"
        ):
            mark_progress("model_setup", True)
            st.balloons()
            st.success(
                "Excellent! Head to the **Tools** page to give your agent superpowers!"
            )

    else:
        st.info("""
        **To proceed, you need to:**
        1. Choose a provider (cloud or local)
        2. Set up your API key or install local tools
        3. Run the health check to verify everything works

        Once you have a working provider, you'll be ready to build amazing agents! 🚀
        """)


if __name__ == "__main__":
    show_model_setup_page()
