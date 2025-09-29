"""
Complete Agent page showing all four components working together in a full agent system.
"""

import streamlit as st
from streamlit_app.components.code_display import (
    display_interactive_code,
    display_code_comparison,
)
from streamlit_app.components.interactive_demo import (
    create_live_agent_chat,
)
from streamlit_app.utils.session_state import mark_progress


def show_complete_agent_page():
    """Display the complete agent page content."""
    from factory import get_recommended_setup, create_tutorial_agent
    from models import get_available_providers, check_all_providers

    st.header("🎯 Complete Agent: Everything Together")

    st.markdown("""
    🎉 **Congratulations!** You've learned about all four core components of AI agents:

    1. **🤖 Intelligence** - Understanding and reasoning
    2. **🛠️ Tools** - Taking actions in the real world
    3. **🧠 Memory** - Maintaining context and structured outputs
    4. **🚦 Routing** - Coordinating multiple specialists

    Now let's see them all working together in a **complete, production-ready agent system**!
    """)

    # Check provider status
    setup = get_recommended_setup()

    if setup["status"] != "ready":
        st.error("⚠️ Please configure a provider first (go to Model Setup page)")
        return

    # The four components working together
    st.markdown("## 🎼 The Four Components in Harmony")

    st.markdown("""
    A complete agent system seamlessly integrates all components:
    """)

    component_integration = [
        {
            "component": "🤖 Intelligence",
            "role": "**Decision Making Core**",
            "description": "The agent understands requests, makes decisions, and provides intelligent responses",
            "example": "Analyzes user request: 'Help me plan a budget for my vacation to Japan'"
        },
        {
            "component": "🛠️ Tools",
            "role": "**Action Layer**",
            "description": "Provides capabilities to gather information, perform calculations, and take concrete actions",
            "example": "Uses currency_converter(), flight_search(), hotel_booking(), and budget_calculator() tools"
        },
        {
            "component": "🧠 Memory",
            "role": "**Context Manager**",
            "description": "Maintains conversation history and returns structured data in useful formats",
            "example": "Remembers user preferences and returns a structured vacation budget with categories"
        },
        {
            "component": "🚦 Routing",
            "role": "**Orchestration Layer**",
            "description": "Coordinates multiple specialized agents and manages complex workflows",
            "example": "Routes flight search to Travel Agent, budget calculation to Finance Agent"
        }
    ]

    for comp in component_integration:
        with st.expander(f"{comp['component']}: {comp['role']}", expanded=False):
            st.markdown(f"**Function:** {comp['description']}")
            st.markdown(f"**Example:** {comp['example']}")

    st.markdown("---")

    # Complete agent architecture
    st.markdown("## 🏗️ Complete Agent Architecture")

    complete_agent_code = '''
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing import List, Optional, Annotated
import datetime

# ============ STRUCTURED OUTPUT MODELS ============
class FlightOption(BaseModel):
    airline: str
    price: float
    departure_time: str
    duration: str
    stops: int

class BudgetCategory(BaseModel):
    category: str
    estimated_cost: float
    description: str

class VacationPlan(BaseModel):
    destination: str
    duration_days: int
    total_budget: float
    budget_breakdown: List[BudgetCategory]
    flight_options: List[FlightOption]
    recommendations: List[str]

# ============ SPECIALIZED AGENTS ============
travel_agent = Agent(
    "openai:gpt-4o-mini",
    instructions="""
    You are a travel specialist. Help users find flights, hotels, and activities.
    Always provide specific, actionable recommendations with prices when possible.
    """
)

finance_agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    instructions="""
    You are a financial advisor specializing in travel budgets.
    Create detailed budget breakdowns and money-saving tips.
    """
)

# ============ TOOLS FOR REAL-WORLD ACTIONS ============
@travel_agent.tool
def search_flights(
    origin: Annotated[str, "Departure city"],
    destination: Annotated[str, "Destination city"],
    date: Annotated[str, "Departure date (YYYY-MM-DD)"]
) -> List[FlightOption]:
    """Search for flight options between cities."""
    # In a real implementation, this would call a flight API
    return [
        FlightOption(
            airline="Japan Airlines",
            price=1200.00,
            departure_time="2024-06-15 10:30",
            duration="14h 20m",
            stops=1
        ),
        FlightOption(
            airline="ANA",
            price=1350.00,
            departure_time="2024-06-15 14:45",
            duration="13h 15m",
            stops=0
        )
    ]

@finance_agent.tool
def calculate_budget(
    destination: Annotated[str, "Travel destination"],
    duration: Annotated[int, "Trip duration in days"],
    budget_level: Annotated[str, "Budget level: budget, mid-range, luxury"]
) -> List[BudgetCategory]:
    """Calculate detailed budget breakdown for a trip."""
    # Simplified budget calculation
    daily_rates = {
        "budget": {"accommodation": 50, "food": 30, "activities": 25, "transport": 15},
        "mid-range": {"accommodation": 120, "food": 60, "activities": 50, "transport": 30},
        "luxury": {"accommodation": 300, "food": 150, "activities": 100, "transport": 75}
    }

    rates = daily_rates.get(budget_level, daily_rates["mid-range"])

    return [
        BudgetCategory(
            category="Accommodation",
            estimated_cost=rates["accommodation"] * duration,
            description=f"Hotels/hostels for {duration} nights"
        ),
        BudgetCategory(
            category="Food & Dining",
            estimated_cost=rates["food"] * duration,
            description="Meals and drinks throughout the trip"
        ),
        BudgetCategory(
            category="Activities & Sightseeing",
            estimated_cost=rates["activities"] * duration,
            description="Tours, attractions, and experiences"
        ),
        BudgetCategory(
            category="Local Transportation",
            estimated_cost=rates["transport"] * duration,
            description="Trains, buses, taxis within destination"
        )
    ]

# ============ MASTER COORDINATOR AGENT ============
vacation_planner = Agent(
    "openai:gpt-4o-mini",
    result_type=VacationPlan,
    instructions="""
    You are a vacation planning coordinator that helps users plan complete trips.

    Your workflow:
    1. Understand the user's travel desires and constraints
    2. Coordinate with travel and finance specialists
    3. Gather flight options and budget breakdowns
    4. Synthesize everything into a comprehensive vacation plan

    Always return a complete VacationPlan with all required fields.
    Be specific, practical, and helpful in your recommendations.
    """
)

# ============ ROUTING AND ORCHESTRATION ============
async def plan_complete_vacation(user_request: str) -> VacationPlan:
    """
    Complete vacation planning workflow that demonstrates all 4 components:
    - Intelligence: Understanding user needs and making decisions
    - Tools: Searching flights and calculating budgets
    - Memory: Maintaining context and returning structured data
    - Routing: Coordinating multiple specialized agents
    """

    # Intelligence: Analyze the user request
    analysis = await vacation_planner.run(f"""
        Analyze this vacation request and extract key details:

        User Request: {user_request}

        Extract:
        - Destination
        - Duration (estimate if not specified)
        - Budget preferences (estimate level if not specified)
        - Any special requirements

        Then coordinate the planning process.
    """)

    # Memory: The agent maintains context throughout the conversation
    # Tools: The vacation_planner will automatically use the available tools
    # Routing: The system coordinates between different specialized agents

    return analysis.output

# ============ USAGE EXAMPLE ============
async def demo_complete_agent():
    """Demonstrate the complete agent system in action"""

    user_request = """
    I want to plan a 7-day vacation to Japan for next summer.
    I'm interested in culture and food, with a mid-range budget.
    I'm traveling from San Francisco.
    """

    # The complete agent system handles everything
    vacation_plan = await plan_complete_vacation(user_request)

    print(f"Destination: {vacation_plan.destination}")
    print(f"Duration: {vacation_plan.duration_days} days")
    print(f"Total Budget: ${vacation_plan.total_budget}")

    print("\\nBudget Breakdown:")
    for category in vacation_plan.budget_breakdown:
        print(f"- {category.category}: ${category.estimated_cost}")

    print("\\nFlight Options:")
    for flight in vacation_plan.flight_options:
        print(f"- {flight.airline}: ${flight.price} ({flight.duration})")
'''

    display_interactive_code(
        complete_agent_code,
        "This complete agent system demonstrates all four components working together:",
        allow_edit=False
    )

    st.markdown("---")

    # Interactive complete agent demo
    st.markdown("## 🎮 Try the Complete Agent System")

    st.markdown("""
    Experience all four components working together! This demo simulates a complete
    vacation planning agent that uses intelligence, tools, memory, and routing.
    """)

    # Demo suggestions
    demo_suggestions = [
        "Plan a 5-day budget trip to Paris for art and culture",
        "I need a luxury 10-day honeymoon in Bali with beach resorts",
        "Family vacation to Orlando with kids, mid-range budget, 7 days",
        "Business trip to Tokyo, 3 days, need efficiency and convenience",
        "Adventure backpacking trip through Thailand, 2 weeks, budget travel"
    ]

    st.markdown("**Try these vacation planning requests:**")

    for i, suggestion in enumerate(demo_suggestions):
        if st.button(f"🏖️ \"{suggestion}\"", key=f"vacation_suggestion_{i}"):
            with st.spinner("🤖 Complete agent system working..."):
                # Simulate the complete agent workflow
                import time
                time.sleep(2)  # Simulate processing time

                # Mock a structured response showing all components
                st.success("✅ Complete vacation plan generated!")

                # Show the workflow steps
                st.markdown("**🔄 Workflow Steps Completed:**")

                workflow_steps = [
                    "🧠 Intelligence: Analyzed request and extracted key requirements",
                    "🚦 Routing: Coordinated with Travel Agent and Finance Agent",
                    "🛠️ Tools: Searched flights and calculated budget breakdown",
                    "🧠 Memory: Synthesized all information into structured plan"
                ]

                for step in workflow_steps:
                    st.markdown(f"- {step}")

                # Mock structured output
                st.markdown("**📊 Structured Output Generated:**")

                mock_plan = {
                    "destination": suggestion.split("to ")[-1].split(" ")[0] if "to " in suggestion else "Destination",
                    "duration_days": 5 if "5-day" in suggestion else 7,
                    "total_budget": 2500.00,
                    "components_used": {
                        "Intelligence": "Request analysis and decision making",
                        "Tools": "Flight search and budget calculation",
                        "Memory": "Context preservation and structured output",
                        "Routing": "Multi-agent coordination"
                    }
                }

                st.json(mock_plan)

    # Manual input for complete agent
    st.markdown("**Or describe your own vacation plans:**")

    user_vacation_request = st.text_area(
        "Describe your ideal vacation:",
        placeholder="I want to plan a trip to...",
        key="vacation_request"
    )

    if st.button("🎯 Plan My Vacation", key="plan_vacation") and user_vacation_request:
        with st.spinner("🤖 Complete agent system analyzing your request..."):
            import time
            time.sleep(3)  # Simulate complex processing

            st.success("✅ Your complete vacation plan is ready!")

            # Show all four components in action
            st.markdown("### 🎼 All Four Components Working Together")

            components_demo = [
                {
                    "component": "🤖 Intelligence",
                    "action": f"Analyzed your request: '{user_vacation_request[:50]}...'",
                    "result": "Identified key requirements: destination, budget, activities, timeline"
                },
                {
                    "component": "🚦 Routing",
                    "action": "Coordinated between specialist agents",
                    "result": "Travel Agent → flight search, Finance Agent → budget planning"
                },
                {
                    "component": "🛠️ Tools",
                    "action": "Executed multiple tools automatically",
                    "result": "search_flights(), calculate_budget(), get_weather(), find_activities()"
                },
                {
                    "component": "🧠 Memory",
                    "action": "Maintained context and structured the output",
                    "result": "Generated complete VacationPlan with all details organized"
                }
            ]

            for comp in components_demo:
                with st.expander(f"{comp['component']}: {comp['action']}", expanded=True):
                    st.markdown(f"**Result:** {comp['result']}")

            # Final structured output
            st.markdown("### 📋 Your Vacation Plan")

            final_plan = {
                "request_analyzed": user_vacation_request,
                "plan_generated": True,
                "all_components_used": True,
                "ready_for_booking": True
            }

            st.json(final_plan)

    st.markdown("---")

    # Real-world complete agent examples
    st.markdown("## 🌍 Real-World Complete Agent Systems")

    real_world_examples = [
        {
            "name": "🩺 Healthcare AI Assistant",
            "intelligence": "Understands medical symptoms and context",
            "tools": "Access medical databases, schedule appointments, check drug interactions",
            "memory": "Maintains patient history and treatment plans",
            "routing": "Routes to specialists: cardiologist, dermatologist, general practitioner"
        },
        {
            "name": "💰 Financial Planning Bot",
            "intelligence": "Analyzes financial goals and risk tolerance",
            "tools": "Portfolio analysis, market data, tax calculations, investment research",
            "memory": "Tracks financial history and generates structured reports",
            "routing": "Coordinates investment advisor, tax specialist, retirement planner"
        },
        {
            "name": "🎓 Educational Tutor System",
            "intelligence": "Adapts to learning style and comprehension level",
            "tools": "Access curriculum databases, generate practice problems, grade assignments",
            "memory": "Tracks progress and maintains personalized learning path",
            "routing": "Routes to subject experts: math tutor, science teacher, writing coach"
        },
        {
            "name": "🏢 Business Operations AI",
            "intelligence": "Understands business processes and optimization opportunities",
            "tools": "CRM integration, inventory management, financial reporting, task automation",
            "memory": "Maintains business context and generates operational insights",
            "routing": "Coordinates sales agent, support agent, finance agent, HR agent"
        }
    ]

    for example in real_world_examples:
        with st.expander(example["name"], expanded=False):
            st.markdown(f"**🤖 Intelligence:** {example['intelligence']}")
            st.markdown(f"**🛠️ Tools:** {example['tools']}")
            st.markdown(f"**🧠 Memory:** {example['memory']}")
            st.markdown(f"**🚦 Routing:** {example['routing']}")

    st.markdown("---")

    # Building your own complete agent
    st.markdown("## 🛠️ Building Your Own Complete Agent")

    st.markdown("""
    Now that you understand all four components, here's your roadmap for building
    production-ready agent systems:
    """)

    building_steps = [
        {
            "step": "1. 🎯 Define Your Use Case",
            "description": "Clearly identify what problems your agent will solve",
            "checklist": [
                "What tasks will users ask for?",
                "What data sources do you need?",
                "What actions should the agent take?",
                "How complex are the workflows?"
            ]
        },
        {
            "step": "2. 🤖 Design the Intelligence Layer",
            "description": "Choose models and design the core reasoning system",
            "checklist": [
                "Select appropriate model providers (OpenAI, Anthropic, local)",
                "Write clear system instructions",
                "Design error handling and fallbacks",
                "Plan for different conversation styles"
            ]
        },
        {
            "step": "3. 🛠️ Implement Tools and Actions",
            "description": "Build the capabilities your agent needs",
            "checklist": [
                "Identify required external APIs and data sources",
                "Create type-safe tool functions with proper validation",
                "Add error handling and retry logic",
                "Test tool reliability and performance"
            ]
        },
        {
            "step": "4. 🧠 Design Memory and Data Flow",
            "description": "Plan how information flows through your system",
            "checklist": [
                "Define conversation context management",
                "Design structured output formats with Pydantic",
                "Plan data persistence strategy",
                "Implement context window management"
            ]
        },
        {
            "step": "5. 🚦 Implement Routing and Orchestration",
            "description": "Coordinate multiple agents and complex workflows",
            "checklist": [
                "Identify specialized agent roles",
                "Design routing logic (rule-based or AI-powered)",
                "Implement workflow coordination",
                "Add load balancing and fallback mechanisms"
            ]
        },
        {
            "step": "6. 🧪 Test and Iterate",
            "description": "Validate your complete system works reliably",
            "checklist": [
                "Test individual components in isolation",
                "Test end-to-end workflows with real scenarios",
                "Monitor performance and error rates",
                "Gather user feedback and iterate"
            ]
        }
    ]

    for step_info in building_steps:
        with st.expander(step_info["step"], expanded=False):
            st.markdown(step_info["description"])
            st.markdown("**Checklist:**")
            for item in step_info["checklist"]:
                st.markdown(f"- {item}")

    st.markdown("---")

    # Success metrics
    st.markdown("## 📊 Measuring Agent Success")

    success_metrics = [
        "**Task Completion Rate**: What percentage of user requests are successfully fulfilled?",
        "**User Satisfaction**: Are users happy with the quality and relevance of responses?",
        "**Response Time**: How quickly does your agent provide useful answers?",
        "**Tool Usage Effectiveness**: Are tools being called appropriately and successfully?",
        "**Error Handling**: How gracefully does the system handle unexpected situations?",
        "**Cost Efficiency**: Are you optimizing for model usage and API costs?",
        "**Scalability**: Can your system handle increased load and complexity?",
        "**Maintainability**: How easy is it to add new capabilities and fix issues?"
    ]

    st.markdown("**Key metrics for production agent systems:**")
    for metric in success_metrics:
        st.markdown(f"- {metric}")

    st.markdown("---")

    # Congratulations and next steps
    st.markdown("## 🎉 Congratulations!")

    st.markdown("""
    You've completed the comprehensive AI Agents tutorial! You now understand:

    ✅ **The Four Core Components:**
    - 🤖 Intelligence: Understanding and reasoning
    - 🛠️ Tools: Taking actions in the real world
    - 🧠 Memory: Context and structured outputs
    - 🚦 Routing: Multi-agent coordination

    ✅ **How They Work Together:**
    - Seamless integration in production systems
    - Real-world applications across industries
    - Best practices for building reliable agents

    ✅ **Practical Implementation:**
    - PydanticAI framework for type-safe development
    - Multiple model provider support
    - Production-ready patterns and architectures
    """)

    # What's next?
    st.markdown("## 🚀 What's Next?")

    next_steps = [
        "🛠️ **Build Your First Agent**: Start with a simple use case and gradually add complexity",
        "📚 **Explore PydanticAI Documentation**: Deep dive into advanced features and patterns",
        "🌐 **Try Different Model Providers**: Experiment with OpenAI, Anthropic, HuggingFace, and local models",
        "🔧 **Integrate with Your Applications**: Connect agents to your existing systems and workflows",
        "👥 **Join the Community**: Share your creations and learn from other agent builders",
        "📈 **Scale to Production**: Implement monitoring, logging, and reliability patterns",
        "🎯 **Specialize Your Agents**: Create domain-specific agents for your industry or use case"
    ]

    for step in next_steps:
        st.markdown(f"- {step}")

    st.markdown("---")

    # Resources and links
    st.markdown("## 📖 Additional Resources")

    resources = [
        "[PydanticAI Documentation](https://pydantic-ai.pydantic.dev/) - Official framework documentation",
        "[Agent Examples Repository](https://github.com/pydantic/pydantic-ai) - Real-world agent implementations",
        "[Model Provider Documentation](https://pydantic-ai.pydantic.dev/models/) - Integration guides",
        "[Community Discord](https://discord.gg/pydantic) - Connect with other developers",
        "[Agent Design Patterns](https://pydantic-ai.pydantic.dev/patterns/) - Production-ready patterns"
    ]

    for resource in resources:
        st.markdown(f"- {resource}")

    # Final progress button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("🎯 Tutorial Complete!", key="tutorial_complete", type="primary"):
            mark_progress("complete_agent", True)
            st.balloons()
            st.success("🎉 Congratulations! You're now ready to build amazing AI agents!")

    st.markdown("---")

    # Thank you message
    st.info("""
    🙏 **Thank you for completing the AI Agents tutorial!**

    You're now equipped with the knowledge to build intelligent, capable agents that can
    understand, act, remember, and coordinate. The future of AI is in your hands!

    Happy agent building! 🤖✨
    """)


if __name__ == "__main__":
    show_complete_agent_page()