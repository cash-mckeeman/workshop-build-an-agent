"""
Routing page for learning about advanced agent orchestration and multi-agent workflows.
"""

import streamlit as st

from src.streamlit_app.components.code_display import (
    display_interactive_code,
)
from src.streamlit_app.utils.session_state import mark_progress


def show_routing_page():
    """Display the routing page content."""
    from src.factory import get_recommended_setup

    st.header("🚦 Routing & Agent Orchestration")

    st.markdown("""
    **Routing** is the fourth core component of advanced agent systems. It's about coordination:
    - **Multiple specialized agents** working together
    - **Intelligent routing** of requests to the right agent
    - **Workflow orchestration** for complex multi-step tasks
    - **Decision-making** about which tools or agents to use

    This transforms simple agents into sophisticated AI systems!
    """)

    # Check provider status
    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("⚠️ Please configure a provider first (go to Model Setup page)")
        return

    # What is routing?
    st.markdown("## 🎯 What is Agent Routing?")

    st.markdown("""
    Think of routing like a smart receptionist at a company who knows exactly which expert to send you to:
    - **Coding questions** → Programming expert
    - **Math problems** → Math specialist
    - **Creative writing** → Creative writing assistant
    - **Complex tasks** → Coordinator who manages multiple experts
    """)

    # Single vs multi-agent comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Single Agent (Does Everything)**")
        single_agent_example = """
# One agent tries to handle everything
general_agent = Agent("openai:gpt-4o-mini",
    instructions="Help with any task"
)

# User: "Write code and explain math"
# Agent: Mediocre at both coding and math
# ❌ Jack of all trades, master of none
"""
        st.code(single_agent_example, language="python")

    with col2:
        st.markdown("**Multi-Agent Router (Specialists)**")
        multi_agent_example = """
# Specialized agents for different tasks
router = Agent("openai:gpt-4o-mini",
    instructions="Route requests to specialists"
)

coding_expert = Agent("openai:gpt-4o-mini",
    instructions="Python coding specialist"
)

math_expert = Agent("anthropic:claude-3-haiku",
    instructions="Mathematics expert"
)

# Router decides: coding → coding_expert
#                 math → math_expert
# ✅ Each expert excels in their domain
"""
        st.code(multi_agent_example, language="python")

    st.markdown("---")

    # Routing patterns
    st.markdown("## ⚙️ Common Routing Patterns")

    routing_patterns = [
        {
            "name": "🎯 Intent-Based Routing",
            "description": "Route based on what the user wants to accomplish",
            "example": "Coding help → Programming Agent, Math help → Math Agent",
            "code": '''
def route_by_intent(user_message):
    """Route based on user intent detection"""
    if "code" in user_message or "python" in user_message:
        return coding_agent
    elif "math" in user_message or "calculate" in user_message:
        return math_agent
    elif "write" in user_message or "story" in user_message:
        return creative_agent
    else:
        return general_agent
''',
        },
        {
            "name": "📊 Capability-Based Routing",
            "description": "Route based on required capabilities",
            "example": "Needs web search → Web Agent, Needs file access → File Agent",
            "code": '''
def route_by_capability(required_capabilities):
    """Route based on required tools/capabilities"""
    agents = {
        "web_search": web_agent,
        "file_operations": file_agent,
        "calculations": math_agent,
        "image_generation": creative_agent
    }

    for capability in required_capabilities:
        if capability in agents:
            return agents[capability]

    return general_agent
''',
        },
        {
            "name": "🔄 Sequential Workflow Routing",
            "description": "Route through multiple agents in sequence",
            "example": "Research → Analysis → Writing → Review",
            "code": '''
async def sequential_workflow(user_request):
    """Multi-step workflow with different agents"""

    # Step 1: Research
    research = await research_agent.run(user_request)

    # Step 2: Analysis
    analysis = await analysis_agent.run(
        f"Analyze this research: {research.output}"
    )

    # Step 3: Writing
    final_result = await writing_agent.run(
        f"Write a report based on: {analysis.output}"
    )

    return final_result
''',
        },
        {
            "name": "🤖 AI-Powered Smart Routing",
            "description": "Use an AI agent to decide routing",
            "example": "Router agent analyzes request and picks best specialist",
            "code": '''
router_agent = Agent("openai:gpt-4o-mini",
    instructions="""
    You are a routing agent. Analyze user requests and decide
    which specialist agent should handle them:
    - "coding": Programming tasks
    - "math": Mathematical problems
    - "creative": Creative writing
    - "research": Information gathering

    Respond with just the agent name.
    """
)

async def smart_route(user_message):
    """AI-powered routing decision"""
    route_decision = await router_agent.run(
        f"Which agent should handle: {user_message}"
    )

    agent_map = {
        "coding": coding_agent,
        "math": math_agent,
        "creative": creative_agent,
        "research": research_agent
    }

    chosen_agent = agent_map.get(route_decision.output, general_agent)
    return await chosen_agent.run(user_message)
''',
        },
    ]

    for pattern in routing_patterns:
        with st.expander(
            f"{pattern['name']}: {pattern['description']}", expanded=False
        ):
            st.markdown(f"**Example:** {pattern['example']}")
            st.code(pattern["code"], language="python")

    st.markdown("---")

    # Interactive routing demo
    st.markdown("## 🎮 Interactive Routing Demo")

    st.markdown("""
    Let's see intelligent routing in action! This demo simulates a multi-agent system
    where different specialists handle different types of requests.
    """)

    # Simulated routing demo
    demo_requests = [
        {
            "request": "Help me write a Python function to calculate compound interest",
            "expected_route": "Programming Agent",
            "reasoning": "Contains 'Python function' - clearly a coding task",
        },
        {
            "request": "What's the derivative of x² + 3x + 5?",
            "expected_route": "Math Agent",
            "reasoning": "Mathematical calculus question",
        },
        {
            "request": "Write a creative story about a robot learning to paint",
            "expected_route": "Creative Agent",
            "reasoning": "Creative writing request",
        },
        {
            "request": "Research the latest trends in artificial intelligence",
            "expected_route": "Research Agent",
            "reasoning": "Information gathering and research task",
        },
    ]

    st.markdown("**Try these example requests to see routing in action:**")

    for i, demo in enumerate(demo_requests):
        with st.expander(f"Example {i + 1}: {demo['request'][:50]}...", expanded=False):
            st.markdown(f"**Full Request:** {demo['request']}")
            st.markdown(f"**Expected Route:** {demo['expected_route']}")
            st.markdown(f"**Reasoning:** {demo['reasoning']}")

            if st.button("🔀 Test Routing", key=f"route_demo_{i}"):
                with st.spinner("Analyzing request and routing..."):
                    # Simulate routing logic
                    import time

                    time.sleep(1)  # Simulate processing

                    st.success(f"✅ Routed to: **{demo['expected_route']}**")
                    st.info(f"🧠 Router reasoning: {demo['reasoning']}")

    # Manual routing test
    st.markdown("**Or test with your own request:**")
    user_request = st.text_input(
        "Enter a request to see how it would be routed:", key="manual_routing"
    )

    if st.button("🔀 Route My Request", key="manual_route") and user_request:
        with st.spinner("Analyzing request..."):
            # Simulate intelligent routing
            routing_logic = {
                "code": ["Programming Agent", "Contains programming-related keywords"],
                "python": ["Programming Agent", "Python-specific coding request"],
                "function": ["Programming Agent", "Programming function request"],
                "math": ["Math Agent", "Mathematical problem"],
                "calculate": ["Math Agent", "Calculation required"],
                "derivative": ["Math Agent", "Calculus operation"],
                "write": ["Creative Agent", "Writing or creative task"],
                "story": ["Creative Agent", "Creative writing request"],
                "creative": ["Creative Agent", "Creative task"],
                "research": ["Research Agent", "Information gathering needed"],
                "find": ["Research Agent", "Information search required"],
                "trends": ["Research Agent", "Current information needed"],
            }

            # Simple routing based on keywords
            route = "General Agent"
            reasoning = "Default general purpose handling"

            for keyword, (agent, reason) in routing_logic.items():
                if keyword.lower() in user_request.lower():
                    route = agent
                    reasoning = reason
                    break

            st.success(f"✅ Would route to: **{route}**")
            st.info(f"🧠 Reasoning: {reasoning}")

    st.markdown("---")

    # Multi-agent coordination
    st.markdown("## 🎼 Multi-Agent Coordination")

    st.markdown("""
    Advanced systems don't just route to single agents - they coordinate multiple agents
    working together on complex tasks.
    """)

    coordination_example = '''
from pydantic import BaseModel
from typing import List

class WorkflowStep(BaseModel):
    agent: str
    task: str
    input: str
    output: str

class WorkflowResult(BaseModel):
    steps: List[WorkflowStep]
    final_result: str

async def complex_workflow(user_request: str) -> WorkflowResult:
    """Coordinate multiple agents for complex tasks"""

    workflow_steps = []

    # Step 1: Planning Agent analyzes the request
    planning_result = await planning_agent.run(
        f"Break down this complex request into steps: {user_request}"
    )

    workflow_steps.append(WorkflowStep(
        agent="Planning Agent",
        task="Analyze and break down request",
        input=user_request,
        output=planning_result.output
    ))

    # Step 2: Research Agent gathers information
    research_result = await research_agent.run(
        f"Research information for: {planning_result.output}"
    )

    workflow_steps.append(WorkflowStep(
        agent="Research Agent",
        task="Gather relevant information",
        input=planning_result.output,
        output=research_result.output
    ))

    # Step 3: Analysis Agent processes the data
    analysis_result = await analysis_agent.run(
        f"Analyze this research data: {research_result.output}"
    )

    workflow_steps.append(WorkflowStep(
        agent="Analysis Agent",
        task="Process and analyze data",
        input=research_result.output,
        output=analysis_result.output
    ))

    # Step 4: Writing Agent creates final output
    writing_result = await writing_agent.run(
        f"Write a comprehensive response based on: {analysis_result.output}"
    )

    workflow_steps.append(WorkflowStep(
        agent="Writing Agent",
        task="Create final response",
        input=analysis_result.output,
        output=writing_result.output
    ))

    return WorkflowResult(
        steps=workflow_steps,
        final_result=writing_result.output
    )

# Usage example:
# result = await complex_workflow("Create a business plan for an AI startup")
'''

    display_interactive_code(
        coordination_example,
        "This workflow coordinates 4 different agents to handle complex requests:",
        allow_edit=False,
    )

    st.markdown("---")

    # Real-world routing examples
    st.markdown("## 🌍 Real-World Routing Examples")

    real_world_examples = [
        {
            "name": "📚 Educational Platform",
            "description": "Route students to subject-specific tutors",
            "agents": [
                "Math Tutor Agent - Handles algebra, calculus, geometry",
                "Science Tutor Agent - Physics, chemistry, biology",
                "Language Arts Agent - Writing, literature, grammar",
                "History Agent - World history, civics, social studies",
                "Coordinator Agent - Manages learning paths and progress",
            ],
        },
        {
            "name": "🏥 Healthcare Assistant",
            "description": "Route medical queries to appropriate specialists",
            "agents": [
                "Symptom Checker Agent - Initial symptom analysis",
                "Medication Agent - Drug interactions and information",
                "Exercise Agent - Fitness and wellness recommendations",
                "Mental Health Agent - Stress and wellness support",
                "Triage Agent - Determines urgency and routing",
            ],
        },
        {
            "name": "💼 Business Automation",
            "description": "Route business tasks to specialized departments",
            "agents": [
                "Sales Agent - Customer inquiries and lead qualification",
                "Support Agent - Technical support and troubleshooting",
                "Finance Agent - Billing, invoicing, and payments",
                "HR Agent - Employee questions and policy information",
                "Executive Agent - Strategic decisions and reporting",
            ],
        },
        {
            "name": "🛍️ E-commerce Platform",
            "description": "Route customer requests through shopping journey",
            "agents": [
                "Product Agent - Search and recommendations",
                "Inventory Agent - Stock levels and availability",
                "Pricing Agent - Discounts and dynamic pricing",
                "Shipping Agent - Delivery options and tracking",
                "Returns Agent - Refunds and exchange processing",
            ],
        },
    ]

    for example in real_world_examples:
        with st.expander(
            f"{example['name']}: {example['description']}", expanded=False
        ):
            st.markdown("**Specialized Agents:**")
            for agent in example["agents"]:
                st.markdown(f"- {agent}")

    st.markdown("---")

    # Advanced routing concepts
    st.markdown("## 🚀 Advanced Routing Concepts")

    advanced_concepts = [
        {
            "title": "⚖️ Load Balancing",
            "description": "Distribute requests across multiple instances of the same agent",
            "code": """
import random
from typing import List

class LoadBalancer:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.current_loads = [0] * len(agents)

    def route_request(self, request: str) -> Agent:
        # Find agent with lowest current load
        min_load_index = self.current_loads.index(min(self.current_loads))

        # Increment load counter
        self.current_loads[min_load_index] += 1

        return self.agents[min_load_index]
""",
        },
        {
            "title": "🔄 Fallback Routing",
            "description": "Automatic fallback when primary agents fail",
            "code": '''
async def fallback_routing(request: str, primary_agent: Agent,
                          fallback_agents: List[Agent]) -> str:
    """Try primary agent, fallback to others if it fails"""

    try:
        result = await primary_agent.run(request)
        return result.output
    except Exception as e:
        print(f"Primary agent failed: {e}")

        for fallback in fallback_agents:
            try:
                result = await fallback.run(request)
                return result.output
            except Exception:
                continue

        return "All agents failed to handle the request"
''',
        },
        {
            "title": "📊 Adaptive Routing",
            "description": "Learn and improve routing decisions over time",
            "code": '''
class AdaptiveRouter:
    def __init__(self):
        self.success_rates = {}
        self.route_history = []

    def route_with_learning(self, request: str, available_agents: dict) -> Agent:
        # Use success rates to pick best agent
        best_agent = None
        best_score = 0

        for agent_name, agent in available_agents.items():
            score = self.success_rates.get(agent_name, 0.5)  # Start at 50%
            if score > best_score:
                best_score = score
                best_agent = agent

        # Record routing decision
        self.route_history.append((request, best_agent))
        return best_agent

    def update_success_rate(self, agent_name: str, success: bool):
        """Update success rates based on feedback"""
        current_rate = self.success_rates.get(agent_name, 0.5)
        # Simple learning algorithm
        if success:
            self.success_rates[agent_name] = min(1.0, current_rate + 0.1)
        else:
            self.success_rates[agent_name] = max(0.0, current_rate - 0.1)
''',
        },
    ]

    for concept in advanced_concepts:
        with st.expander(concept["title"], expanded=False):
            st.markdown(concept["description"])
            st.code(concept["code"], language="python")

    st.markdown("---")

    # Routing best practices
    st.markdown("## 💡 Routing Best Practices")

    best_practices = [
        "**Clear specialization**: Each agent should have a well-defined domain of expertise",
        "**Fallback strategies**: Always have backup plans when primary routing fails",
        "**Performance monitoring**: Track which routes work best for different request types",
        "**Load management**: Distribute work evenly across agent instances",
        "**Error handling**: Gracefully handle failures in multi-agent workflows",
        "**User feedback**: Allow users to correct routing decisions to improve the system",
        "**Context preservation**: Maintain conversation context across agent handoffs",
        "**Cost optimization**: Route expensive requests to powerful models, simple ones to efficient models",
    ]

    for practice in best_practices:
        st.markdown(f"- {practice}")

    st.markdown("---")

    # When to use routing
    st.markdown("## 🤔 When to Use Routing")

    routing_scenarios = [
        {
            "scenario": "✅ **Use Routing When:**",
            "examples": [
                "You have clearly different types of requests (coding vs writing vs math)",
                "Tasks require specialized knowledge or capabilities",
                "You want to optimize cost (route simple tasks to cheaper models)",
                "You need different response formats or styles for different use cases",
                "Complex multi-step workflows require coordination",
            ],
        },
        {
            "scenario": "❌ **Skip Routing When:**",
            "examples": [
                "All requests are similar in nature",
                "A single general-purpose agent handles everything well",
                "The overhead of routing isn't worth the specialization benefits",
                "You're building a simple prototype or MVP",
                "Latency is critical and routing adds too much delay",
            ],
        },
    ]

    for scenario in routing_scenarios:
        st.markdown(f"### {scenario['scenario']}")
        for example in scenario["examples"]:
            st.markdown(f"- {example}")

    st.markdown("---")

    # Key takeaways
    st.markdown("## 🎯 Key Takeaways")

    st.markdown("""
    **What you've learned:**
    - ✅ Routing coordinates multiple specialized agents
    - ✅ Different patterns: intent-based, capability-based, workflow-based
    - ✅ AI-powered routing can make intelligent decisions
    - ✅ Multi-agent coordination enables complex workflows
    - ✅ Real-world systems use sophisticated routing strategies

    **The power of routing:**
    - 🎯 **Specialization**: Each agent excels in their domain
    - 🔄 **Coordination**: Multiple agents work together seamlessly
    - 📈 **Scalability**: Add new specialists without rebuilding everything
    - 💰 **Cost efficiency**: Route based on complexity and requirements
    - 🛡️ **Reliability**: Fallback mechanisms prevent single points of failure
    """)

    # Progress button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button(
            "✅ I Understand Routing!", key="understand_routing", type="primary"
        ):
            mark_progress("routing", True)
            st.success(
                "Amazing! Now see everything work together in the **Complete Agent** demo!"
            )

    st.markdown("---")

    st.info("""
    🚀 **Next up:** See all four components working together in a **Complete Agent** -
    intelligence, tools, memory, and routing combined into a powerful AI system!
    """)


if __name__ == "__main__":
    show_routing_page()
