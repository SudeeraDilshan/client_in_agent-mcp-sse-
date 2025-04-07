import os
from enum import Enum
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

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    CombinedMemory
)

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

from langdetect import detect
from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agents_tools import inbuilt_tools

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

langchain.debug = False


class AgentState(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    ROUTING = "routing"
    COLLECTING_INFO = "collecting_info"
    OFFLINE = "offline"


class EnhancedConversationalAgent(BaseAgent):

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
                description="Retrieves relevant context from the knowledge base for customer queries."
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

        # Initialize memory components
        self.memory_config = {
            "max_token_limit": 4000,  # Token limit for buffer memory
            "window_size": 5,  # Number of recent interactions to retain
            "summary_interval": 10,  # Interactions between summaries
        }

        self.language_map = {
            "english": "English",
            "spanish": "Spanish",
            "español": "Spanish",
            "french": "French",
            "français": "French",
            "german": "German",
            "deutsch": "German",
            "chinese": "Chinese",
            "mandarin": "Chinese",
            "italian": "Italian",
            "italiano": "Italian",
            "japanese": "Japanese",
            "日本語": "Japanese"
        }

        self.department_map = {
            "sales": "sales",
            "purchase": "sales",
            "buy": "sales",
            "pricing": "sales",
            "customer service": "customer_service",
            "support": "customer_service",
            "help": "customer_service",
            "account": "customer_service",
            "billing": "customer_service",
            "technical": "technical_support",
            "tech": "technical_support",
            "IT": "technical_support",
            "technical support": "technical_support",
            "product": "technical_support",
            "bug": "technical_support",
            "issue": "technical_support"
        }

        self.routing_chain = self._create_routing_chain()
        self.conversation_chain = self._create_conversation_chain()

    async def connect_to_mcp_server(self, server_url="http://localhost:8000/sse"):
        """Connect to the MCP server and load tools."""
        try:
            # Store the client as instance attribute to prevent garbage collection
            self.mcp_sse_client = sse_client(server_url)
            streams = await self.mcp_sse_client.__aenter__()
            self.mcp_server_session = ClientSession(streams[0], streams[1])
            await self.mcp_server_session.__aenter__()
            await self.mcp_server_session.initialize()

            # Load MCP tools
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
        - Only use tools when necessary to fulfill a customer request that requires external data or actions.
        - Be sure to use the RAG tool to retrieve information from our knowledge base when answering product or service questions.
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

        # Create an OpenAI functions agent with the tools
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        # Create an agent executor with more robust error handling
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            # handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True,
            # max_execution_time=30  # Add timeout to prevent hanging
        )
        
        def process_result(result):
            if isinstance(result, dict) and "intermediate_steps" in result:
             # Process intermediate steps for MCP tool outputs
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        tool_output = step[1]
                        if isinstance(tool_output, list) and len(tool_output) > 0 and hasattr(tool_output[0], "text"):
                            result["output"] = "\n".join([item.text for item in tool_output if hasattr(item, "text")])
                        break
            return result

        return RunnablePassthrough() | agent_executor | process_result

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

            # Special handling for ROUTING and COLLECTING_INFO states
            if current_state == AgentState.ROUTING:
                return self._handle_routing_state(context, data, session_memory)
            elif current_state == AgentState.COLLECTING_INFO:
                return self._handle_collecting_info_state(context, data, session_memory)

            # Check if we need human routing based on explicit requests
            if self._needs_human_interaction(data):
                return self._handle_human_routing(context, data, session_memory)

            # Use the agent chain directly for all tool selection and conversation handling
            chain_input = {
                "input": data,
                "chat_history": chat_history
            }

            try:
                # Invoke the agent chain to handle tool selection
                chain_result = await self.agent_chain.ainvoke(chain_input)

                output = ""
                
                # Check if we have a dictionary result from the chain
                if isinstance(chain_result, dict):
                    if "output" in chain_result:
                        output = chain_result["output"]
                    elif "return_values" in chain_result:
                        output = chain_result["return_values"]
                    # Check if we have intermediate steps that might contain tool outputs
                    elif "intermediate_steps" in chain_result:
                        steps = chain_result["intermediate_steps"]
                        for step in steps:
                            if len(step) >= 2:  # Each step should be (action, output)
                                tool_output = step[1]
                                # Handle MCP tool outputs specifically
                                if isinstance(tool_output, list) and len(tool_output) > 0:
                                    if hasattr(tool_output[0], "text"):
                                        # For MCP tools that return TextContent objects
                                        output += "\n".join([item.text for item in tool_output if hasattr(item, "text")])
                                    else:
                                        # For other list-type outputs
                                        output += "\n".join([str(item) for item in tool_output])
                                else:
                                    output += str(tool_output)
                # Direct string output
                elif isinstance(chain_result, str):
                    output = chain_result
                # Handle direct MCP tool outputs
                elif isinstance(chain_result, list) and len(chain_result) > 0:
                    if hasattr(chain_result[0], "text"):
                        output = "\n".join([item.text for item in chain_result if hasattr(item, "text")])
                    else:
                        output = "\n".join([str(item) for item in chain_result])
                else:
                    # Last resort - convert whatever we got to string
                    output = str(chain_result)
                
                print(f"Agent output: {output}")

                # Always return output wrapped in AgentOutput object
                return AgentOutput(result=output)

            except Exception as context_error:
                print(f"Error in agent chain: {str(context_error)}")
                # Fall back to conversation chain if agent chain fails
                try:
                    # Prepare input for conversation chain
                    memory_system = session_memory.get("memory_system")
                    conversation_input = {"input": data}

                    # Add memory components to the input
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

                    # Use conversation chain as fallback
                    output = await self.conversation_chain.ainvoke(conversation_input)
                except Exception as fallback_error:
                    print(f"Context window exceeded: {str(fallback_error)}")
                    output = self._handle_context_window_exceeded(
                        session_id, data)

            # Update chat history
            session_memory["chat_history"].extend(
                [HumanMessage(content=data), AIMessage(content=output)]
            )

            # Update memory with this interaction
            self._update_memory(session_id, data, output)

            # Update context
            self._update_context(context, session_memory)

            # Always return the result wrapped in AgentOutput
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

    def _create_memory_system(self, session_id: str) -> Dict:
        """Create a combined memory system using different memory types."""

        # Recent messages memory (keeps last few interactions)
        buffer_window_memory = ConversationBufferWindowMemory(
            k=self.memory_config["window_size"],
            return_messages=True,
            output_key="output",
            input_key="input",
            memory_key="recent_messages"
        )

        # Token-limited memory for managing context length
        token_buffer_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=self.memory_config["max_token_limit"],
            return_messages=True,
            output_key="output",
            input_key="input",
            memory_key="token_buffer"
        )

        # Summary memory for long conversations
        summary_memory = ConversationSummaryMemory(
            llm=self.summary_llm,
            return_messages=True,
            output_key="output",
            input_key="input",
            memory_key="conversation_summary"
        )

        # Combined memory system
        memory_system = {
            "buffer_window": buffer_window_memory,
            "token_buffer": token_buffer_memory,
            "summary": summary_memory,
            "message_count": 0,
            "session_id": session_id
        }

        return memory_system

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
        - Customer Service: Address account issues, general inquiries, and service problems.
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

    def _extract_language(self, text: str) -> Optional[str]:
        """Extract language preference from text using improved detection."""
        text_lower = text.lower()

        for keyword, language in self.language_map.items():
            if keyword in text_lower:
                return language

        try:
            lang_code = detect(text)

            iso_to_language = {
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'zh-cn': 'Chinese',
                'zh-tw': 'Chinese',
                'zh': 'Chinese',
                'it': 'Italian',
                'ja': 'Japanese'
            }

            if lang_code in iso_to_language:
                return iso_to_language[lang_code]
        except:
            pass

        return None

    def _extract_department(self, text: str) -> Optional[str]:
        """Extract department preference from text using improved detection."""
        text_lower = text.lower()

        for keyword, department in self.department_map.items():
            if keyword in text_lower:
                return department

        return None

    def _needs_human_interaction(self, input_text: str) -> bool:
        """Determine if the customer needs to be routed to a human agent."""
        human_keywords = [
            # English
            "talk to human", "speak to agent", "human agent",
            "real person", "customer service", "representative",
            "live person", "human support", "live support",
            "human help", "speak with someone", "transfer me",
            "connect me", "speak to a person", "talk to a representative"
            # Spanish
            "hablar con humano", "hablar con agente", "persona real",
            "servicio al cliente", "representante", "ayuda humana",
            # French
            "parler à un humain", "parler à un agent", "personne réelle",
            "service client", "représentant", "aide humaine",
            # German
            "mit einem menschen sprechen", "mit einem agenten sprechen",
            "echter mensch", "kundendienst", "vertreter", "menschliche hilfe",
            # Simple patterns that work across languages
            "human", "agent", "person", "humano", "agente", "persona",
            "humain", "mensch", "representative", "representante"
        ]

        if any(keyword in input_text.lower() for keyword in human_keywords):
            return True

        frustration_indicators = [
            "this is ridiculous", "not helpful", "useless",
            "waste of time", "frustrated", "annoyed",
            "speak to your manager", "unhelpful", "incompetent"
        ]

        if any(indicator in input_text.lower() for indicator in frustration_indicators):
            return True

        return False

    def _handle_human_routing(self, context, data, session_memory):
        """Handle routing to a human agent with automatic department detection."""

        session_id = session_memory["routing_info"]["session_id"]
        room_id = session_memory["routing_info"]["room"]
        routing_info = session_memory["routing_info"]

        # First, determine the user's language preference
        language = routing_info.get('language')
        if not language:
            detected_language = self._extract_language(data)
            if detected_language:
                language = detected_language
            else:
                # Default to English if we can't detect
                language = "English"

        # Then, try to determine the department
        department = routing_info.get('department')
        if not department:
            # Try to extract from current message
            department = self._extract_department(data)

            # If not found in current message, analyze conversation history
            if not department:
                department = self._analyze_conversation_for_department(
                    session_id)

                if not department:
                    routing_info['language'] = language
                routing_info['query'] = data
                session_memory["state"] = AgentState.COLLECTING_INFO

                # Create a response asking about the department
                response_message = f"I'll connect you with a human agent. Which department do you need? Options: Sales, Customer Service, or Technical Support."

                # Update conversation history and memory
                session_memory["chat_history"].extend(
                    [HumanMessage(content=data), AIMessage(
                        content=response_message)]
                )
                self._update_memory(session_id, data, response_message)
                self._update_context(context, session_memory)

                return AgentOutput(
                    result=response_message,
                    metadata={"action": "collecting_info",
                              "missing_info": "department"}
                )

        # Now we have both language and department, route to human
        routing_result = self._route_to_human(
            data, language, department, session_id, room_id
        )

        # Update routing info and state
        routing_info['language'] = language
        routing_info['department'] = department
        routing_info['human_connected'] = True
        session_memory["state"] = AgentState.ROUTING

        # Create response message
        response_message = routing_result.get(
            'message', f"Connecting you to our {department} team who speaks {language}. A representative will assist you shortly."
        )

        # Update conversation history and memory
        session_memory["chat_history"].extend(
            [HumanMessage(content=data), AIMessage(content=response_message)]
        )
        self._update_memory(session_id, data, response_message)
        self._update_context(context, session_memory)

        return AgentOutput(
            result=response_message,
            metadata={"action": "human_handoff",
                      "routing_result": routing_result}
        )

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

    def _get_session_memory(self, session_id: str):
        """Retrieve or create session memory."""
        # In a production environment, this would likely use Redis or another persistent store
        # For now, we'll use a class-level dictionary to simulate persistence
        if not hasattr(self, "_memory_store"):
            self._memory_store = {}

        if session_id not in self._memory_store:
            self._memory_store[session_id] = self._create_memory_system(
                session_id)

        return self._memory_store[session_id]

    def _update_memory(self, session_id: str, human_message: str, ai_message: str) -> None:
        """Update the memory system with new conversation turns."""
        memory_system = self._get_session_memory(session_id)

        # Update all memory components
        for memory_key in ["buffer_window", "token_buffer"]:
            memory = memory_system[memory_key]
            memory.save_context({"input": human_message},
                                {"output": ai_message})

        # Increment message counter
        memory_system["message_count"] += 1

        # Check if it's time to update the summary
        if memory_system["message_count"] % self.memory_config["summary_interval"] == 0:
            self._update_conversation_summary(session_id)

    def _update_conversation_summary(self, session_id: str) -> None:
        """Update the conversation summary based on token buffer memory."""
        memory_system = self._get_session_memory(session_id)

        # Get messages from token buffer
        token_buffer = memory_system["token_buffer"]
        buffer_messages = token_buffer.buffer

        if not buffer_messages:
            return

        # Create a summary input with the buffer messages
        chat_history = buffer_messages
        summary_input = {"chat_history": chat_history}

        try:
            # Generate a new summary
            summary = self._create_summary_chain().invoke(summary_input)

            # Save the summary to summary memory
            summary_memory = memory_system["summary"]
            summary_memory.clear()  # Clear old summary
            summary_memory.save_context(
                {"input": "Conversation history"}, {"output": summary})
        except Exception as e:
            print(f"Error updating conversation summary: {str(e)}")

    def _handle_context_window_exceeded(self, session_id: str, input_text: str):
        """Handle cases where the context window is exceeded."""
        memory_system = self._get_session_memory(session_id)

        try:
            # First try to reduce the token buffer size
            token_buffer = memory_system["token_buffer"]
            token_buffer.max_token_limit = token_buffer.max_token_limit // 2

            # Force a summary update
            self._update_conversation_summary(session_id)

            # Use only the summary and most recent messages
            fallback_input = {
                "input": input_text,
                "conversation_summary": memory_system["summary"].buffer
            }

            # Use the fallback chain
            response = self._create_fallback_chain().invoke(fallback_input)

            return response
        except Exception as e:
            print(f"Fallback mechanism failed: {str(e)}")
            self._reset_memory(session_id)
            return "I apologize, but I seem to have lost track of our conversation. Could you please summarize what you were asking about so I can help you better?"

    def _reset_memory(self, session_id: str) -> None:
        """Reset memory for a session as a last resort fallback."""
        if not hasattr(self, "_memory_store"):
            return

        if session_id in self._memory_store:
            del self._memory_store[session_id]

    def _initialize_session_memory(self, context):
        """Initialize session memory with enhanced memory components."""
        session_id = context.session_id if hasattr(
            context, 'session_id') else "default_session"

        # Initialize or retrieve memory system
        memory_system = self._get_session_memory(session_id)

        # Set up session information
        session_memory = {
            "chat_history": getattr(context, 'chat_history', []),
            "state": getattr(context, 'agent_state', AgentState.AVAILABLE),
            "routing_info": getattr(context, 'routing_info', {
                "language": None,
                "department": None,
                "query": None,
                "session_id": session_id,
                "room": context.request.get("room_id", "default_room"),
                "human_connected": False
            }),
            "memory_system": memory_system
        }

        return session_memory

    def _update_context(self, context, session_memory):
        """Update context with session memory."""
        context.add_to_context("chat_history", session_memory["chat_history"])
        context.add_to_context("agent_state", session_memory["state"])
        context.add_to_context("routing_info", session_memory["routing_info"])

        # Add memory components to context if needed
        if "memory_system" in session_memory:
            context.add_to_context(
                "memory_system", session_memory["memory_system"])

    def _handle_routing_state(self, context, data, session_memory):
        """Handle the ROUTING state with enhanced memory integration."""
        routing_info = session_memory["routing_info"]
        chat_history = session_memory["chat_history"]
        session_id = routing_info["session_id"]
        room_id = routing_info["room"]

        if not routing_info.get('human_connected', False):
            language = routing_info.get('language') or "English"
            department = routing_info.get('department') or "customer_service"
            query = routing_info.get('query') or data

            # Route to human with enhanced context from memories
            routing_result = self._route_to_human(
                query, language, department, session_id, room_id)
            routing_info['human_connected'] = True
            response_message = routing_result.get(
                'message', f"Connecting you to {department} in {language}...")

            chat_history.extend(
                [HumanMessage(content=data), AIMessage(content=response_message)])

            # Update memory with this interaction
            self._update_memory(session_id, data, response_message)

            session_memory["chat_history"] = chat_history
            session_memory["routing_info"] = routing_info
            self._update_context(context, session_memory)

            return AgentOutput(result=response_message, metadata={"action": "human_handoff", "routing_result": routing_result})
        else:
            forward_message = "Your message has been forwarded to the human agent."
            chat_history.extend(
                [HumanMessage(content=data), AIMessage(content=forward_message)])

            # Update memory even for forwarded messages
            self._update_memory(session_id, data, forward_message)

            session_memory["chat_history"] = chat_history
            self._update_context(context, session_memory)

            return AgentOutput(result=forward_message, metadata={"action": "forward_to_human", "message": data, "session_id": session_id, "room_id": room_id})

    def _handle_collecting_info_state(self, context, data, session_memory):
        """Handle the COLLECTING_INFO state with enhanced memory integration."""
        routing_info = session_memory["routing_info"]
        chat_history = session_memory["chat_history"]
        session_id = routing_info["session_id"]

        memory_system = session_memory.get("memory_system")
        routing_input = {'input': data, 'chat_history': chat_history}

        if memory_system and "buffer_window" in memory_system:
            recent_messages = memory_system["buffer_window"].load_memory_variables({
            })
            if "recent_messages" in recent_messages:
                routing_input["recent_messages"] = recent_messages["recent_messages"]

        # Extract language and department from message
        detected_language = self._extract_language(data)
        detected_department = self._extract_department(data)

        if not detected_language and "recent_messages" in routing_input:
            for message in routing_input["recent_messages"]:
                if isinstance(message, HumanMessage):
                    lang = self._extract_language(message.content)
                    if lang:
                        detected_language = lang
                        break

        if not detected_language:
            detected_language = "English"

        if detected_language and not routing_info.get('language'):
            routing_info['language'] = detected_language
        if detected_department and not routing_info.get('department'):
            routing_info['department'] = detected_department

        # Check if we have enough information to route
        if routing_info.get('language') and routing_info.get('department'):
            session_memory["state"] = AgentState.ROUTING
            self._update_context(context, session_memory)
            return self._handle_routing_state(context, data, session_memory)

        # Build response message
        info_message = f"I've noted your preference"
        if detected_language:
            info_message += f" for {detected_language} language"
        if detected_language and detected_department:
            info_message += " and"
        if detected_department:
            info_message += f" for {detected_department} department"
        info_message += "."

        if not routing_info.get('department'):
            info_message += " Which department do you need assistance with? Options: Sales, Customer Service, Technical Support."

        # Update memory and context
        chat_history.extend([HumanMessage(content=data),
                            AIMessage(content=info_message)])
        self._update_memory(session_id, data, info_message)

        session_memory["chat_history"] = chat_history
        session_memory["routing_info"] = routing_info
        self._update_context(context, session_memory)

        return AgentOutput(result=info_message)

    def _analyze_conversation_for_department(self, session_id: str) -> Optional[str]:
        """Analyze the entire conversation history to determine the most likely department needed."""

        # Get memory system and conversation history
        memory_system = self._get_session_memory(session_id)

        # Start with an empty set of department-related keywords
        department_keywords = set()

        # Collect all user messages from different memory components
        user_messages = []

        # Get messages from recent window memory
        if "buffer_window" in memory_system:
            recent_messages = memory_system["buffer_window"].load_memory_variables(
                {}).get("recent_messages", [])
            for message in recent_messages:
                if isinstance(message, HumanMessage):
                    user_messages.append(message.content)

        # Also check token buffer for longer history
        if "token_buffer" in memory_system:
            token_messages = memory_system["token_buffer"].load_memory_variables(
                {}).get("token_buffer", [])
            for message in token_messages:
                if isinstance(message, HumanMessage):
                    user_messages.append(message.content)

        # Combine all user messages into one text for analysis
        combined_text = " ".join(user_messages).lower()

        # Count occurrences of department-related keywords
        department_counts = {
            "sales": 0,
            "customer_service": 0,
            "technical_support": 0
        }

        # Process the combined text to count department indicators
        for keyword, department in self.department_map.items():
            if keyword in combined_text:
                department_counts[department] += 1

        # If we have a clear winner, return that department
        max_count = max(department_counts.values())
        if max_count > 0:
            # Find department with max count
            for dept, count in department_counts.items():
                if count == max_count:
                    return dept

        # If no clear indicators, fall back to more sophisticated analysis
        return self._advanced_department_analysis(combined_text)

    def _advanced_department_analysis(self, text: str) -> Optional[str]:
        """Use more sophisticated analysis to determine department when keyword matching fails."""

        # Technical support indicators
        tech_patterns = [
            "not working", "broken", "error", "bug", "fix", "problem",
            "troubleshoot", "install", "update", "upgrade", "version",
            "crash", "slow", "login issue", "password reset", "access"
        ]

        # Sales indicators
        sales_patterns = [
            "price", "cost", "discount", "purchase", "buy", "package",
            "subscription", "plan", "trial", "demo", "quote", "offer",
            "availability", "order", "delivery"
        ]

        # Customer service indicators
        cs_patterns = [
            "account", "bill", "charge", "refund", "cancel", "subscription",
            "policy", "complaint", "feedback", "service", "assistance",
            "help", "support", "information", "question"
        ]

        # Count matches for each category
        tech_count = sum(1 for pattern in tech_patterns if pattern in text)
        sales_count = sum(1 for pattern in sales_patterns if pattern in text)
        cs_count = sum(1 for pattern in cs_patterns if pattern in text)

        # Determine the highest count
        counts = {
            "technical_support": tech_count,
            "sales": sales_count,
            "customer_service": cs_count
        }

        max_count = max(counts.values())

        # If we have matches, return the department with the highest count
        if max_count > 0:
            for dept, count in counts.items():
                if count == max_count:
                    return dept

        # Default to customer service if no clear indicators
        return "customer_service"

    
