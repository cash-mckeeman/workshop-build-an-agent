"""
IT Helpdesk Agent Implementation using PydanticAI

This module provides a specialized IT support agent that demonstrates domain-specific
agent behavior with tailored prompting and retrieval strategies for IT support scenarios.
"""

import asyncio
from typing import Optional
import logging

from pydantic_ai import Agent

from ..retrieval.vector_store import FAISSVectorStore
from ..retrieval.reranker import RerankerProvider
from ..tools.retrieval_tools import register_retrieval_tools

logger = logging.getLogger(__name__)


def create_it_helpdesk_agent(
    model_provider: str,
    vector_store: FAISSVectorStore,
    reranker: Optional[RerankerProvider] = None,
    instructions: Optional[str] = None
) -> Agent[None, str]:
    """Create an IT helpdesk agent with specialized support capabilities.

    This agent is specifically designed for IT support scenarios, with prompting
    and behavior optimized for technical troubleshooting and user assistance.

    Args:
        model_provider: Model identifier (e.g., "openai:gpt-4o-mini")
        vector_store: Vector store containing IT documentation and procedures
        reranker: Optional reranker for improving retrieval quality
        instructions: Custom instructions for the agent

    Returns:
        A configured IT helpdesk agent
    """
    default_instructions = """You are an expert IT helpdesk agent providing technical support to users.

Your responsibilities include:
1. Troubleshooting technical issues with computers, software, and systems
2. Providing step-by-step solutions and procedures
3. Explaining technical concepts in user-friendly language
4. Escalating complex issues when necessary
5. Following company policies and security guidelines

When handling support requests:
- Always search the knowledge base first for relevant procedures, troubleshooting steps, and policies
- Provide clear, actionable instructions broken down into steps
- Ask clarifying questions when the issue is unclear
- Prioritize user safety and data security
- Reference specific documentation or procedures when available
- If you can't resolve an issue, clearly explain escalation options

Communication style:
- Be professional, patient, and empathetic
- Use clear, non-technical language unless the user is technically oriented
- Confirm understanding and check if the solution worked
- Provide additional resources or next steps when helpful

Always search the knowledge base using your tools to ensure accurate, up-to-date information."""

    # Create the agent with IT-specific instructions
    agent = Agent(
        model_provider,
        instructions=instructions or default_instructions
    )

    # Register retrieval tools
    register_retrieval_tools(
        agent=agent,
        vector_store=vector_store,
        reranker=reranker,
        include_advanced=True  # IT agents benefit from advanced search capabilities
    )

    logger.info("Created IT helpdesk agent with retrieval tools")
    return agent


async def demo_it_helpdesk_agent(model_provider: Optional[str] = None):
    """Demonstrate the IT helpdesk agent functionality.

    This shows how a specialized agent works with domain-specific knowledge
    to provide technical support and troubleshooting assistance.
    """
    print("🛠️  IT Helpdesk Agent Demo")
    print("=" * 40)

    try:
        # Create embedding provider and components
        from ..retrieval.embeddings import get_best_available_provider
        from ..retrieval.vector_store import create_vector_store
        from ..retrieval.reranker import get_best_available_provider as get_best_reranker
        from ..retrieval.base import Document

        embedding_provider = get_best_available_provider()
        if not embedding_provider:
            print("❌ No embedding provider available")
            return

        # Create vector store with IT knowledge base
        vector_store = create_vector_store(embedding_provider)
        reranker = get_best_reranker()

        # Sample IT knowledge base documents
        it_documents = [
            Document(
                content="""Password Reset Procedure:
                1. Verify user identity by asking for employee ID and department
                2. Check if account is locked in Active Directory
                3. Reset password using admin console: Admin > User Management > Reset Password
                4. Generate temporary password following complexity requirements (8+ chars, mixed case, numbers, symbols)
                5. Email temporary password to user's manager for secure delivery
                6. User must change password on first login
                7. Document the reset in the helpdesk ticket system""",
                metadata={"topic": "password", "category": "procedure", "priority": "standard"}
            ),
            Document(
                content="""VPN Connection Troubleshooting:
                Common issues and solutions:
                - Check internet connectivity first
                - Verify VPN client is latest version
                - Ensure firewall isn't blocking VPN (ports 443, 1723, 500, 4500)
                - Clear DNS cache: ipconfig /flushdns
                - Reset network adapter: network settings > reset network
                - If still failing, check server status and try alternate VPN server
                - For persistent issues, escalate to network team with error logs""",
                metadata={"topic": "vpn", "category": "troubleshooting", "priority": "high"}
            ),
            Document(
                content="""Software Installation Guidelines:
                All software installations must follow security policy:
                1. Check approved software list first
                2. For new software, submit request to security team
                3. Download only from official vendor websites
                4. Run security scan before installation
                5. Install with standard user privileges when possible
                6. Document installation in asset management system
                7. Unapproved software installations are prohibited and may result in disciplinary action""",
                metadata={"topic": "software", "category": "policy", "priority": "standard"}
            ),
            Document(
                content="""Email Setup for Mobile Devices:
                For corporate email access:
                1. Ensure device meets minimum security requirements
                2. Install approved email app (Outlook mobile recommended)
                3. Server settings: mail.company.com, port 993 (IMAP) or Exchange
                4. Enable two-factor authentication
                5. Configure automatic lock screen (max 5 minutes)
                6. Enable remote wipe capability
                7. User must agree to mobile device policy
                8. Test email sync and calendar integration""",
                metadata={"topic": "email", "category": "setup", "priority": "standard"}
            ),
            Document(
                content="""Printer Connection Issues:
                Step-by-step troubleshooting:
                1. Check physical connections (USB/network cable)
                2. Verify printer is powered on and not showing errors
                3. Check if printer appears in device manager
                4. Update or reinstall printer drivers from manufacturer website
                5. Remove and re-add printer in Windows: Settings > Printers & Scanners
                6. For network printers, verify IP address and ping test
                7. Clear print spooler: services.msc > Print Spooler > restart
                8. If still not working, try different USB port or cable""",
                metadata={"topic": "printer", "category": "troubleshooting", "priority": "low"}
            ),
            Document(
                content="""System Performance Optimization:
                When computers are running slowly:
                1. Check CPU and memory usage in Task Manager
                2. Look for high-usage processes and unnecessary startup programs
                3. Run disk cleanup to free up space (need 15% free minimum)
                4. Check for malware using approved antivirus scan
                5. Update Windows and drivers to latest versions
                6. Consider SSD upgrade for older systems with spinning drives
                7. If still slow, may need hardware upgrade consultation
                8. Always backup important data before major changes""",
                metadata={"topic": "performance", "category": "troubleshooting", "priority": "standard"}
            )
        ]

        vector_store.add_documents(it_documents)
        print(f"📚 Loaded {len(it_documents)} IT procedures into knowledge base")

        # Create IT helpdesk agent
        if not model_provider:
            # Try to get best available model
            try:
                from ..models.providers import get_best_available_provider as get_best_model
                model_provider = get_best_model()
            except:
                pass
            if not model_provider:
                model_provider = "openai:gpt-4o-mini"  # fallback

        agent = create_it_helpdesk_agent(model_provider, vector_store, reranker)
        print(f"🤖 Created IT helpdesk agent with model: {model_provider}")

        # Test IT support scenarios
        support_scenarios = [
            "I can't log into my computer. It says my password is incorrect but I'm sure it's right.",
            "The VPN won't connect. I get an error message that says 'connection timeout'.",
            "I need to install a new graphics design software for my project. What's the process?",
            "My laptop is running very slowly and takes forever to start up. What can I do?",
            "I can't print from my computer. The printer shows up but nothing happens when I try to print."
        ]

        for i, scenario in enumerate(support_scenarios, 1):
            print(f"\n{'='*60}")
            print(f"Support Ticket #{i}: {scenario}")
            print("=" * 60)

            try:
                result = await agent.run(scenario)
                print(f"🛠️  IT Support Response:")
                print(result.data)

            except Exception as e:
                print(f"❌ Error processing support request: {str(e)}")
                continue

        print(f"\n{'='*40}")
        print("✅ IT Helpdesk Agent Demo Complete")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.exception("IT helpdesk agent demo failed")


def demo_helpdesk_agent():
    """Synchronous wrapper for the IT helpdesk agent demo."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we're in a loop, we need to use a different approach
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, demo_it_helpdesk_agent())
            future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        asyncio.run(demo_it_helpdesk_agent())


if __name__ == "__main__":
    demo_helpdesk_agent()