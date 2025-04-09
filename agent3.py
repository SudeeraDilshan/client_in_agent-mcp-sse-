import os
from typing import List, Dict, Any, Optional, Union

from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import langchain
from human_tool import HumanRoutingTool
from rag_tool import RAGTool
from human_handling import HumanHandlingMixin, AgentState


from langchain.memory.chat_message_histories import RedisChatMessageHistory

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)

from dialdeskai.src.agents.base import BaseAgent
from dialdeskai.src.agents.config import AgentConfig
from dialdeskai.src.agents.output import AgentOutput

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agents_tools import inbuilt_tools

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from memory import Memory

load_dotenv()

langchain.debug = False


class EnhancedConversationalAgent(BaseAgent, HumanHandlingMixin, Memory):

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        self.human_routing_tool = HumanRoutingTool()
        self.rag_tool = RAGTool()
        self.magic_function_tool = inbuilt_tools[0]
        self.mcp_tools = None
        self.mcp_sse_client = None
        self.mcp_server_session = None

        self.tools = [
            Tool(
                name="human_routing",
                func=self.human_routing_tool.run,
                description="Routes a customer query to a human agent with the appropriate language and department skills"
            ),
            Tool(
                name="rag",
                func=self.rag_tool.get_relevant_context,
                description="Use this tool to answer questions about company products, services, policies, or procedures. It searches our knowledge base for relevant information. Always use this tool when answering questions about our company, products, or services."
            ),
            Tool(
                name="magic_function",
                func=self.magic_function_tool.run,
                description="Adds 10 to any input number. Input should be a valid number (integer or decimal)."
            )
        ]

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        self.summary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1
        )

        self.agent_chain = None

        self.memory_config = {
            "max_token_limit": 4000,
            "window_size": 5,
            "summary_interval": 10,
        }

        self.routing_chain = self._create_routing_chain()
        self.conversation_chain = self._create_conversation_chain()

    async def connect_to_mcp_server(self, server_url="http://localhost:8000/sse"):
        """Connect to the MCP server and load tools."""
        try:
            self.mcp_sse_client = sse_client(server_url)
            streams = await self.mcp_sse_client.__aenter__()
            self.mcp_server_session = ClientSession(streams[0], streams[1])
            await self.mcp_server_session.__aenter__()
            await self.mcp_server_session.initialize()

            self.mcp_tools = await load_mcp_tools(self.mcp_server_session)

            self.tools = self.tools + self.mcp_tools

            self.agent_chain = self._create_agent_chain()

            return True
        except Exception as e:
            print(f"Error connecting to MCP server: {str(e)}")
            if self.agent_chain is None:
                self.agent_chain = self._create_agent_chain()
            return False

    def _create_agent_chain(self) -> Runnable:
        """Create an agent chain that can use tools and handle conversation."""

        system_prompt = """
        You are an AI-powered customer service assistant designed to handle customer inquiries professionally and efficiently.
        
        ## GUIDELINES:
        - Keep responses brief and concise (ideally 2-4 sentences when possible)
        - Use simple, clear language that's easily understood when heard rather than read
        - Avoid complex sentences, jargon, or technical terminology unless necessary
        - Structure responses with the most important information first
        - Use a warm, conversational tone appropriate for spoken dialogue
        
        ## TOOLS USAGE:
        - You have access to several tools to help customers. Always use the most appropriate tool for the task.
        - For ANY questions about company products, services, policies, or procedures, you MUST use the RAG tool to search our knowledge base.
        - The RAG tool is your primary source of information about our company. Use it before providing any information about our products or services.
        - Only use other tools when necessary to fulfill a customer request that requires external data or actions.
        - If the customer wants to speak to a human, use the human_routing tool to connect them to the right department.
        
        ## RESPONSE FORMAT:
        1. Start with a direct answer to the customer's question
        2. Provide only necessary supporting details or context
        3. End with a clear next step or question if appropriate
        
        Remember to prioritize customer satisfaction while efficiently addressing their needs.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            # handle_parsing_errors=True,
            max_iterations=3,
            # return_intermediate_steps=True,
            # max_execution_time=30
        )

        return agent_executor

    async def process_request(self, context):
        """Main request processing method with tool selection."""
        if self.agent_chain is None:
            self.agent_chain = self._create_agent_chain()
        session_memory = self._initialize_session_memory(context)
        data = context.request.get("query")
        if not data:
            return AgentOutput(result="Sorry, I didn't receive any message to process. Please try again.")

        session_id = session_memory["routing_info"]["session_id"]
        chat_history = session_memory["chat_history"]

        try:
            current_state = session_memory["state"]

            if current_state == AgentState.ROUTING:
                return self._handle_routing_state(context, data, session_memory)
            elif current_state == AgentState.COLLECTING_INFO:
                return self._handle_collecting_info_state(context, data, session_memory)

            if self._needs_human_interaction(data):
                return self._handle_human_routing(context, data, session_memory)

            chain_input = {
                "input": data,
                "chat_history": chat_history
            }

            try:
                chain_result = await self.agent_chain.ainvoke(chain_input)

                output = chain_result.get("output", "")

                if isinstance(output, list) and output and hasattr(output[0], 'text'):
                    output = "\n".join(
                        [item.text for item in output if hasattr(item, 'text')])

                print(f"Agent output: {output}")
                return AgentOutput(result=output)

            except Exception as context_error:
                print(f"Error in agent chain: {str(context_error)}")
                try:
                    memory_system = session_memory.get("memory_system")
                    conversation_input = {"input": data}

                    if memory_system:
                        if "buffer_window" in memory_system:
                            recent_vars = memory_system["buffer_window"].load_memory_variables({
                            })
                            if "recent_messages" in recent_vars:
                                conversation_input["recent_messages"] = recent_vars["recent_messages"]

                        if "token_buffer" in memory_system:
                            token_vars = memory_system["token_buffer"].load_memory_variables({
                            })
                            if "token_buffer" in token_vars:
                                conversation_input["token_buffer"] = token_vars["token_buffer"]

                        if "summary" in memory_system:
                            summary_vars = memory_system["summary"].load_memory_variables({
                            })
                            if "conversation_summary" in summary_vars:
                                conversation_input["conversation_summary"] = summary_vars["conversation_summary"]

                    output = await self.conversation_chain.ainvoke(conversation_input)

                    if isinstance(output, list) and output and hasattr(output[0], 'text'):
                        output = "\n".join(
                            [item.text for item in output if hasattr(item, 'text')])
                except Exception as fallback_error:
                    print(f"Context window exceeded: {str(fallback_error)}")
                    output = self._handle_context_window_exceeded(
                        session_id, data)

            session_memory["chat_history"].extend(
                [HumanMessage(content=data), AIMessage(content=output)]
            )

            self._update_memory(session_id, data, output)

            self._update_context(context, session_memory)

            return AgentOutput(result=output)

        except Exception as e:
            error_msg = "I apologize, but I encountered an issue processing your request. Please try again."
            print(f"Error processing input: {str(e)}")

            try:
                self._update_memory(session_id, data, error_msg)
            except:
                pass

            session_memory["chat_history"].extend(
                [HumanMessage(content=data), AIMessage(content=error_msg)]
            )
            self._update_context(context, session_memory)

            return AgentOutput(result=error_msg)

    def _create_conversation_chain(self) -> Runnable:
        """Create the main conversation chain with enhanced prompts and RAG integration."""

        system_prompt = """
        You are an AI-powered customer service assistant designed to handle customer inquiries professionally and efficiently.
        
        ## GUIDELINES:
        - Keep responses brief and concise (ideally 2-4 sentences when possible)
        - Use simple, clear language that's easily understood when heard rather than read
        - Avoid complex sentences, jargon, or technical terminology unless necessary
        - Structure responses with the most important information first
        - Use a warm, conversational tone appropriate for spoken dialogue
        - Pause naturally between different topics using clear transitions
        - When giving instructions, break them into simple, sequential steps
        - Avoid providing unnecessary background information
        - Focus on directly answering the customer's question or addressing their concern
        - If you don't know the answer, admit it clearly and offer to connect the customer to a human agent.
        - Personalize your responses based on the customer's history and preferences when available.
        - Use appropriate language based on the customer's department inquiry and language preference.

        ## RESPONSE FORMAT:
        1. Start with a direct answer to the customer's question
        2. Provide only necessary supporting details or context
        3. End with a clear next step or question if appropriate

        ## KNOWLEDGE BASE:
        - Below is information from our knowledge base that may be relevant to the customer's query.
        - If the information below addresses their question, use it to provide a detailed answer.
        - If the information is not relevant, rely on your general knowledge but make it clear when you're not using specific company information.

        {context}

        ## CONTEXT AWARENESS:
        - Pay attention to the conversation history and avoid asking for information the customer has already provided.
        - Reference previous interactions when relevant to show continuity in the conversation.
        - Maintain awareness of the customer's emotional state and adjust your tone accordingly.
        
        ## ESCALATION PROTOCOL:
        - If the customer explicitly requests to speak with a human agent, initiate the handoff protocol.
        - If the customer appears frustrated or if their query is beyond your capabilities, offer to connect them to a human agent.
        - When escalating, summarize the conversation for the human agent to provide context.
        
        ## DEPARTMENT-SPECIFIC INFORMATION:
        - Sales: Focus on product features, pricing, and purchase process.
        - Customer Service: Address account issues, general inquiries, and non-technical service problems.
        - Technical Support: Help troubleshoot technical issues and provide step-by-step guidance.
        
        Remember to prioritize customer satisfaction while efficiently addressing their needs.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(
                variable_name="conversation_summary", optional=True),
            MessagesPlaceholder(
                variable_name="recent_messages", optional=True),
            MessagesPlaceholder(variable_name="token_buffer", optional=True),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        rag_chain = RunnablePassthrough.assign(
            context=lambda x: self.rag_tool.get_relevant_context(x["input"])
        ) | prompt | self.llm | StrOutputParser()

        return rag_chain

    def _create_routing_chain(self) -> Runnable:
        """Create the routing chain with enhanced prompts."""

        routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
            You are an AI assistant responsible for routing customer inquiries to the appropriate department and in their preferred language.
            
            ## OBJECTIVE:
            Efficiently gather the necessary information to route the customer's inquiry correctly.
            
            ## REQUIRED INFORMATION:
            1. LANGUAGE PREFERENCE: Identify the customer's preferred language from their input. If not specified, politely ask them to choose.
               Available languages: English, Spanish, French, German, Chinese, Italian, Japanese
            
            2. DEPARTMENT NEEDED: Determine which department the customer needs based on their inquiry. If unclear, ask for clarification.
               Available departments:
               - Sales: For purchasing products, pricing inquiries, and sales-related questions
               - Customer Service: For account issues, general inquiries, and non-technical service problems
               - Technical Support: For technical issues, troubleshooting, and product functionality questions
            
            ## GUIDELINES:
            - Ask only for information that hasn't been provided yet.
            - Be concise and direct in your questions.
            - When both language and department are identified, confirm the information and proceed with routing.
            - If the customer expresses urgency, prioritize efficiency in gathering information.
            
            ## EXAMPLE INTERACTIONS:
            
            Example 1:
            Customer: "I want to buy your product."
            You: "I'd be happy to connect you with our Sales department. What language would you prefer for this conversation?"
            
            Example 2:
            Customer: "My account isn't working. I speak Spanish."
            You: "Entendido. Le conectaré con nuestro departamento de Servicio al Cliente en español. Un momento por favor."
            
            Example 3:
            Customer: "I need help with a technical issue in German."
            You: "Ich verbinde Sie mit unserem technischen Support auf Deutsch. Einen Moment bitte."
            """),
            MessagesPlaceholder(
                variable_name="recent_messages", optional=True),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        return routing_prompt | self.llm | StrOutputParser()

    def _route_to_human(self, query: str, language: str, department: str, session_id: str, room_id: str) -> Dict:
        """Route the customer to a human agent with enhanced context."""
        try:

            routing_result = self.human_routing_tool._run(
                query=query,
                language=language,
                department=department,
                sessionId=session_id,
                session=session_id,
                room=room_id,
            )

            if routing_result.get("status") == "success":
                return {
                    "message": f"I'm connecting you with a {department} specialist who speaks {language}. A human agent will assist you shortly."
                }

            return routing_result

        except Exception as e:
            print(f"Error routing to human: {str(e)}")
            return {
                "status": "success",
                "message": f"I'm transferring you to a {department} specialist who speaks {language}. A human agent will assist you shortly."
            }
