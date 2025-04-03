import asyncio
from dialdeskai.src.agents.config import AgentConfig
from dialdeskai.src.runtime.runtime import AgentRuntime
from agent import EnhancedConversationalAgent

async def setup_and_run_agent():
    """ Setup and run the agent with MCP client """
    agent_config = AgentConfig(host='127.0.0.1', port=8080, name='EnhancedAgent', self_register=False)

    # Create agent with MCP client
    agent = await EnhancedConversationalAgent.create(config=agent_config)
    
    try:
        agent_id = AgentRuntime.run_agent(agent)
        print(f"Agent is running with ID: {agent_id}")
        print(f"API documentation available at: http://127.0.0.1:8080/docs")

        AgentRuntime.keep_alive()
    finally:
        await agent.cleanup()

def main():
    """ Main function to run the agent """
    asyncio.run(setup_and_run_agent())

if __name__ == "__main__":
    main()