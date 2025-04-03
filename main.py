import asyncio
from agent import Agent
import traceback

async def main():
    try:
        # Create and initialize agent with MCP client in one step
        agent = Agent()
        await agent.load_mcp_tools({"SSE_URL": "http://localhost:8000/sse"})
        # try:
        await agent.run_interactive()
        # finally:
        #     # The cleanup is now also called inside run_interactive when exiting,
        #     # but we'll keep it here as a safety measure
        #     await agent.cleanup()
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())