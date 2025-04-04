# import asyncio
# from agent import Agent
# import traceback
#
# async def main():
#     agent = None
#     try:
#         agent = Agent()
#         await agent.load_tools(server_url="http://localhost:8000/sse")
#         await agent.run_interactive()
#     except Exception as e:
#         traceback.print_exc()
#         print(f"Error occurred: {e}")
#     finally:
#         # Ensure cleanup is called even if an exception occurs
#         if agent:
#             await agent.cleanup()
#
# if __name__ == "__main__":
#     # Run the async function
#     asyncio.run(main())

import os
from dotenv import load_dotenv
from agent import Agent
from dialdeskai.src.types import AgentConfig, QueueType
from dialdeskai.src.runtime.runtime import AgentRuntime

load_dotenv()

config = AgentConfig(
    name="McpAgent",
    host="localhost",
    port=8000,
    self_register=False,
    queue_config={
        "queue_type": QueueType.MOSQUITTO,
        "connection_params": {"host": "128.199.113.42", "port": 1884, "topic": "DialDeskAI/echo"}
    },
    store_config={
        "store_type": "redis",
        "connection_params": {
            "host": os.getenv('REDIS_HOST'),
            "port": int(os.getenv('REDIS_PORT')),
            "password": os.getenv('REDIS_PASSWORD'),
        }
    },
    postgres_config={
        "host": os.getenv('PSQL_HOST'),
        "port": int(os.getenv('PSQL_PORT')),
        "username": os.getenv('PSQL_USER'),
        "password": os.getenv('PSQL_PASSWORD'),
        "database": os.getenv('PSQL_DB'),
        "use_vector": True
    }
)

async def main():
    try:
        agent = Agent(config)
        await agent.load_tools()
        AgentRuntime.run_agent(agent)
        AgentRuntime.keep_alive()
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
