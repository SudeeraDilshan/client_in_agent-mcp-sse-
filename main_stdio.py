import asyncio
from agent_stdio import Agent
import traceback

async def main():
    agent = None
    try:
        agent = Agent()
        await agent.load_tools()  # Ensure this is awaited
        await agent.run_interactive()  # Ensure this is awaited
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")
    finally:
        if agent and hasattr(agent, 'cleanup'):  # Check if cleanup method exists
            await agent.cleanup()  # Await cleanup if it's a coroutine

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())