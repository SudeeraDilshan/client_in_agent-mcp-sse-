import asyncio
from dialdeskai.src.agents.config import AgentConfig
from dialdeskai.src.runtime.runtime import AgentRuntime
from test import EnhancedConversationalAgent



async def initialize_agent():
    """Initialize and connect the agent to the MCP server"""
    agent_config = AgentConfig(
        host='127.0.0.1', port=8080, name='EnhancedAgent', self_register=False)
    agent = EnhancedConversationalAgent(config=agent_config)
    await agent.connect_to_mcp_server(server_url="http://localhost:8000/sse")
    return agent


def main():
    """ Main function demonstrating the agent's usage """
    # Initialize agent and connect to MCP server
    loop = asyncio.get_event_loop()
    agent = loop.run_until_complete(initialize_agent())

    agent_id = AgentRuntime.run_agent(agent)
    print(f"Agent is running with ID: {agent_id}")
    print(f"API documentation available at: http://127.0.0.1:8080/docs")

    try:
        AgentRuntime.keep_alive()
    except KeyboardInterrupt:
        print("Shutting down...")
        loop.run_until_complete(agent.disconnect_from_mcp_server())


if __name__ == "__main__":
    main()