import os

from tools import inbuilt_tools
from dialdeskai.src.agents.base import BaseAgent
from dialdeskai.src.queue.trigger import EventTrigger
from dialdeskai.src.store.session import Session
from dialdeskai.src.types import AgentOutput, AgentConfig, Message, MessageRole, Event
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from rag import rag_tool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class Agent(BaseAgent):

    def __init__(self, agent_config: AgentConfig, api_key=GOOGLE_API_KEY):
        super().__init__(agent_config)
        self.tools = inbuilt_tools + rag_tool
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("human", "{input}"),
                ("ai", "{agent_scratchpad}"),
            ]
        )
        self.mcp_tools = None
        self.mcp_sse_client = None
        self.mcp_server_session = None
        self.agent = None
        self.agent_executor = None
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )

    async def process_request(self, session: Session) -> AgentOutput:
        data = session.request.data.get("query")
        message = Message(session_id=session.session_id, content=data, role=MessageRole.USER)
        self.add_to_chat_history(message, session_id=session.session_id)

        response = await self.agent_executor.ainvoke({"input": data})
        message = Message(session_id=session.session_id, content=response["output"], role=MessageRole.AGENT)
        self.add_to_chat_history(message, session_id=session.session_id)

        data = {"result": response["output"]}

        event = Event(description="Transformed to uppercase", data=data, session_id=session.session_id)
        EventTrigger.trigger(Event(description="Transformed to uppercase", data=data, session_id=session.session_id))
        session.add_event(event)

        result = AgentOutput(result=data)
        return result

    async def load_tools(self, server_url="http://localhost:8000/sse"):
        self.mcp_sse_client = sse_client(server_url)
        streams = await self.mcp_sse_client.__aenter__()
        self.mcp_server_session = ClientSession(streams[0], streams[1])
        await self.mcp_server_session.__aenter__()
        await self.mcp_server_session.initialize()
        event = Event(description="MCP server session initialized", data={}, sessionless=True)
        EventTrigger.trigger(event)

        # Load MCP tools
        self.mcp_tools = await load_mcp_tools(self.mcp_server_session)
        event = Event(description="MCP tools loaded", data={"tools": self.tools}, sessionless=True)
        EventTrigger.trigger(event)

        self.tools = inbuilt_tools + rag_tool + self.mcp_tools  # Combine inbuilt tools with MCP tools and RAG tool

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        event = Event(description="Langchain tool calling agent created", data={}, sessionless=True)
        EventTrigger.trigger(event)
