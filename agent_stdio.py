from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import inbuilt_tools
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)

class Agent:
    def __init__(self,api_key=GOOGLE_API_KEY):
        self.tools = inbuilt_tools   # Combine the tool lists properly
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
        
    async def load_tools(self):
        server_params = StdioServerParameters(command="env\\Scripts\\python", args=["server_stdio.py"])
        self.stdio_client = stdio_client(server_params)
        streams = await self.stdio_client.__aenter__()
        self.stdio_session = ClientSession(streams[0], streams[1])
        await self.stdio_session.__aenter__()
        await self.stdio_session.initialize()
        
        # Load MCP tools
        self.mcp_tools = await load_mcp_tools(self.stdio_session)
        self.tools = inbuilt_tools + self.mcp_tools
        
        print(self.tools)
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
               
    async def process_input(self, user_input):
        """Process a single user input and return the agent's response"""
        try:
            response = await self.agent_executor.ainvoke({"input": user_input})  # Ensure this is awaited
            return response["output"]  # Access the output correctly
        except Exception as e:
            print(f"Error processing input: {e}")
            return "An error occurred while processing your input."
    
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
            await self.cleanup()  # Ensure cleanup is called

    async def cleanup(self):
        """Cleanup resources for this instance"""
        if hasattr(self, 'stdio_session') and self.stdio_session:
            await self.stdio_session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_client') and self.stdio_client:
            await self.stdio_client.__aexit__(None, None, None)








