import os
# import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from agent_tools import inbuilt_tools  # Custom inbuilt tools

# Load environment variables
load_dotenv()

# Get API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)

class Agent:
    def __init__(self, api_key=GOOGLE_API_KEY):
        self.tools = inbuilt_tools  # Initialize with inbuilt tools
        self.prompt = prompt
        self.mcp_tools = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    # @staticmethod
    # async def create(server_url="http://localhost:8000/sse", api_key=GOOGLE_API_KEY):
    #     """Factory method to create and initialize an agent with MCP client"""
    #     # Initialize the base agent first
    #     agent = Agent(api_key=api_key)  # Using the class name directly
        
    #     # Set up MCP client
    #     agent.sse_client = sse_client(server_url)
    #     agent.streams = await agent.sse_client.__aenter__()
    #     agent.mcp_server_session = ClientSession(agent.streams[0], agent.streams[1])
    #     await agent.mcp_server_session.__aenter__()
    #     await agent.mcp_server_session.initialize()
        
    #     # Load MCP tools
    #     mcp_tools = await load_mcp_tools(agent.mcp_server_session)
    #     agent.tools = inbuilt_tools + mcp_tools
        
    #     # Recreate agent with updated tools
    #     agent.agent = create_tool_calling_agent(agent.llm, agent.tools, agent.prompt)
    #     agent.agent_executor = AgentExecutor(agent=agent.agent, tools=agent.tools, verbose=True)
        
    #     return agent
    
    # async def load_tools(self,mcp_config:dict[str, str] = None):
    #     """Load tools based on the provided configuration"""
    #     if mcp_config:
    #         # Load MCP tools based on the provided configuration
    #         sse_client = sse_client(mcp_config["server_url"])
    #         streams = await sse_client.__aenter__()
    #         self.mcp_server_session = ClientSession(streams[0],streams[1])
    #         await self.mcp_server_session.__aenter__()
    #         await self.mcp_server_session.initialize()
            
    #         mcp_tools = await load_mcp_tools(self.mcp_server_session)
    #         self.tools = inbuilt_tools + mcp_tools
            
    #         # Recreate agent with updated tools
    #         self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
    #         self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    #     else:
    #         # Load inbuilt tools only
    #         self.tools = inbuilt_tools
    #         self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
    #         self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    

    async def load_mcp_tools(self, mcp_config: dict[str, str] = None):
        """Load tools based on the provided configuration"""
        try:
            async with sse_client(mcp_config.get("SSE_URL","http://localhost:8000/sse")) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()

                    mcp_tools = await load_mcp_tools(session)
                    self.tools = inbuilt_tools + mcp_tools
                    
        except Exception as e:
            # traceback.print_exc()
            print(f"Error occurred: {e}")

    
    async def process_input(self, user_input):
        """Process a single user input and return the agent's response"""
        response = await self.agent_executor.ainvoke({"input": user_input})
        return response["output"]
    
    async def run_interactive(self):
        """Run an interactive session with the agent"""
        if self.mcp_tools:
            self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    await self.cleanup()  # Clean up resources before exiting
                    break
                
                response = await self.process_input(user_input)
                print("AI:", response)
            except KeyboardInterrupt:
                await self.cleanup()  # Also clean up on keyboard interrupt
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def cleanup(self):
        """Cleanup resources for this instance"""
        if self.mcp_server_session:
            await self.mcp_server_session.__aexit__(None, None, None)
        if hasattr(self, 'sse_client') and self.sse_client:
            await self.sse_client.__aexit__(None, None, None)








