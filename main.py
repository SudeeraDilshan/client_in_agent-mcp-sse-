# import os
# from dotenv import load_dotenv
# from agent import Agent
# from dialdeskai.src.types import AgentConfig, QueueType
# from dialdeskai.src.runtime.runtime import AgentRuntime

# load_dotenv()

# config = AgentConfig(
#     name="McpAgent",
#     host="localhost",
#     port=8000,
#     self_register=False,
#     queue_config={
#         "queue_type": QueueType.MOSQUITTO,
#         "connection_params": {"host": "128.199.113.42", "port": 1884, "topic": "DialDeskAI/echo"}
#     },
#     store_config={
#         "store_type": "redis",
#         "connection_params": {
#             "host": os.getenv('REDIS_HOST'),
#             "port": int(os.getenv('REDIS_PORT')),
#             "password": os.getenv('REDIS_PASSWORD'),
#         }
#     },
#     postgres_config={
#         "host": os.getenv('PSQL_HOST'),
#         "port": int(os.getenv('PSQL_PORT')),
#         "username": os.getenv('PSQL_USER'),
#         "password": os.getenv('PSQL_PASSWORD'),
#         "database": os.getenv('PSQL_DB'),
#         "use_vector": True
#     }
# )

# async def main():
#     try:
#         agent = Agent(config)
#         await agent.load_tools()
#         AgentRuntime.run_agent(agent)
#         AgentRuntime.keep_alive()
#     except Exception as e:
#         print(f"Error occurred: {e}")


# if __name__ == '__main__':
#     import asyncio
#     asyncio.run(main())


import asyncio
from dialdeskai.src.agents.config import AgentConfig
from dialdeskai.src.runtime.runtime import AgentRuntime
from agent import Agent

async def initialize_agent():
    """Initialize and connect the agent to the MCP server"""
    agent_config = AgentConfig(
        host='127.0.0.1', port=8080, name='EnhancedAgent', self_register=False)
    agent = Agent(config=agent_config)
    
    await agent.load_tools(server_url="http://localhost:8000/sse")
    
    print(agent.tools)
    return agent


async def main():
    """ Main function demonstrating the agent's usage """
    # # Initialize agent and connect to MCP server
    # loop = asyncio.get_event_loop()
    # agent = loop.run_until_complete(initialize_agent())
    
    agent =  await initialize_agent()

    agent_id = AgentRuntime.run_agent(agent)
    print(f"Agent is running with ID: {agent_id}")
    print(f"API documentation available at: http://127.0.0.1:8080/docs")

    try:
        AgentRuntime.keep_alive()
    except KeyboardInterrupt:
        print("Shutting down...")
        # loop.run_until_complete(agent.disconnect_from_mcp_server())


if __name__ == "__main__":
    asyncio.run(main())