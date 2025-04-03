import asyncio
from agent import Agent
import traceback

async def main():
    agent = None
    try:
        # Create and initialize agent with MCP client in one step
        agent = Agent()
        await agent.load_tools(server_url="http://localhost:8000/sse")
        await agent.run_interactive()
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")
    finally:
        # Ensure cleanup is called even if an exception occurs
        if agent:
            await agent.cleanup()

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())