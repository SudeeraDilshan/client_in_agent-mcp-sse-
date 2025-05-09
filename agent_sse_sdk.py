import os
from dialdeskai.src.agents.output import AgentOutput
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from tools import inbuilt_tools  # Custom inbuilt tools
from dialdeskai.src.agents.base import BaseAgent
from dialdeskai.src.agents.config import AgentConfig

# from rag_tool import rag_tool
# from rag2 import rag_tool  # Custom RAG tool

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

class Agent(BaseAgent):
    def __init__(self,config:AgentConfig, api_key=GOOGLE_API_KEY):
        super().__init__(config)
        self.tools = inbuilt_tools  # Initialize with inbuilt tools
        self.prompt = prompt
        self.mcp_tools = None
        self.mcp_sse_client = None
        self.mcp_server_session = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
        print(self.tools)
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    

    async def load_tools(self, server_url="http://localhost:8000/sse"):
        # Store the client as instance attribute to prevent garbage collection
        self.mcp_sse_client = sse_client(server_url)
        streams = await self.mcp_sse_client.__aenter__()
        self.mcp_server_session = ClientSession(streams[0], streams[1])
        await self.mcp_server_session.__aenter__()
        await self.mcp_server_session.initialize()
        
        # Load MCP tools
        self.mcp_tools = await load_mcp_tools(self.mcp_server_session)
        self.tools = self.tools + self.mcp_tools  # Combine inbuilt tools with MCP tools and RAG tool
        
        # Recreate agent with updated tools
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    async def process_input(self, user_input):
        """Process a single user input and return the agent's response"""
        response = await self.agent_executor.ainvoke({"input": user_input})
        return response["output"]
    
    async def run_interactive(self):
        """Run an interactive session with the agent"""
        try:
            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break
                    
                    response = await self.process_input(user_input)
                    print("AI:", response)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
        finally:
            # Clean up resources when exiting
            await self.cleanup()
            
            
    async def process_request(self, context):
        """Process a request with the agent"""
        try:
            data = context.request.get("query")
            response = await self.agent_executor.ainvoke({"input": data})
            output= response["output"]
            return AgentOutput(result=output)
        
        except Exception as e:
            print(f"Error processing request: {e}")
            return None        
    
    async def cleanup(self):
        """Cleanup resources for this instance"""
        if hasattr(self, 'mcp_server_session') and self.mcp_server_session:
            await self.mcp_server_session.__aexit__(None, None, None)
        if hasattr(self, 'sse_client') and self.mcp_sse_client:
            await self.mcp_sse_client.__aexit__(None, None, None)















