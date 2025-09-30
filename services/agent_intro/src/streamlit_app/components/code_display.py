"""
Code display components for the Streamlit app.
"""

from typing import Any

import streamlit as st


def display_interactive_code(
    code: str,
    explanation: str,
    language: str = "python",
    allow_edit: bool = False,
    key: str = None,
) -> str:
    """Display code with syntax highlighting and optional editing."""
    st.markdown(explanation)

    if allow_edit and key:
        edited_code = st.text_area(
            "Edit the code:", value=code, height=200, key=f"code_edit_{key}"
        )
        st.code(edited_code, language=language)
        return edited_code
    else:
        st.code(code, language=language)
        return code


def show_execution_step(
    step_name: str,
    input_data: Any,
    output_data: Any,
    code: str = None,
    explanation: str = None,
):
    """Show step-by-step execution with data flow."""
    with st.expander(f"🔍 {step_name}", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            if isinstance(input_data, dict):
                st.json(input_data)
            else:
                st.write(input_data)

        with col2:
            st.subheader("Output")
            if isinstance(output_data, dict):
                st.json(output_data)
            else:
                st.write(output_data)

        if code:
            st.subheader("Code")
            st.code(code, language="python")

        if explanation:
            st.subheader("Explanation")
            st.markdown(explanation)


def display_code_comparison(
    title1: str, code1: str, title2: str, code2: str, language: str = "python"
):
    """Display two code blocks side by side for comparison."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(title1)
        st.code(code1, language=language)

    with col2:
        st.subheader(title2)
        st.code(code2, language=language)


def display_code_evolution(steps: list[dict[str, str]], current_step: int = -1):
    """Display code evolution through multiple steps."""
    st.subheader("Code Evolution")

    for i, step in enumerate(steps):
        is_current = i == current_step
        is_completed = i < current_step

        if is_current:
            st.markdown(f"**👉 Step {i + 1}: {step['title']}**")
        elif is_completed:
            st.markdown(f"✅ Step {i + 1}: {step['title']}")
        else:
            st.markdown(f"⏳ Step {i + 1}: {step['title']}")

        with st.expander(f"View Step {i + 1} Code", expanded=is_current):
            if step.get("explanation"):
                st.markdown(step["explanation"])
            st.code(step["code"], language=step.get("language", "python"))


def display_runnable_code(
    code: str,
    run_button_text: str = "Run Code",
    explanation: str = None,
    key: str = None,
) -> bool:
    """Display code with a run button."""
    if explanation:
        st.markdown(explanation)

    st.code(code, language="python")

    return st.button(run_button_text, key=key)


def display_tool_signature(
    tool_name: str, parameters: dict[str, Any], return_type: str, description: str
):
    """Display tool function signature and documentation."""
    # Create function signature
    param_strs = []
    for param_name, param_info in parameters.items():
        param_type = param_info.get("type", "Any")
        param_desc = param_info.get("description", "")
        param_strs.append(f"{param_name}: {param_type}")

    signature = f"def {tool_name}({', '.join(param_strs)}) -> {return_type}:"

    with st.expander(f"🔧 {tool_name} Function", expanded=False):
        st.code(signature, language="python")
        st.markdown(f"**Description:** {description}")

        if parameters:
            st.markdown("**Parameters:**")
            for param_name, param_info in parameters.items():
                param_type = param_info.get("type", "Any")
                param_desc = param_info.get("description", "No description")
                st.markdown(f"- `{param_name}` ({param_type}): {param_desc}")


def display_agent_architecture():
    """Display interactive agent architecture diagram."""
    st.subheader("🏗️ Agent Architecture")

    # Create a visual representation of agent components
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **🧠 MODEL**
        - Language Model
        - Reasoning Engine
        - Text Generation
        """)

    with col2:
        st.markdown("""
        **🛠️ TOOLS**
        - Function Calls
        - External APIs
        - Calculations
        """)

    with col3:
        st.markdown("""
        **💾 MEMORY**
        - Chat History
        - Context Window
        - State Tracking
        """)

    with col4:
        st.markdown("""
        **🚦 ROUTING**
        - Decision Logic
        - Tool Selection
        - Flow Control
        """)

    # Show data flow
    st.markdown("---")
    st.markdown("""
    **Data Flow:**
    ```
    User Input → MODEL (understands) → ROUTING (decides) → TOOLS (executes) → MEMORY (stores) → Response
    ```
    """)


def display_code_playground(initial_code: str = "", key: str = "playground") -> str:
    """Display an interactive code playground."""
    st.subheader("🎮 Code Playground")
    st.markdown("Edit and experiment with the code below:")

    # Code editor
    code = st.text_area(
        "Python Code:",
        value=initial_code,
        height=300,
        key=f"playground_{key}",
        help="Write your Python code here. Click 'Run Code' to execute.",
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        run_button = st.button("▶️ Run Code", key=f"run_{key}")

    with col2:
        if run_button:
            st.info("💡 In a real implementation, this would execute the code safely.")

    return code
