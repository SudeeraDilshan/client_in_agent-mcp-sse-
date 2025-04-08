
from enum import Enum
from typing import Dict
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    CombinedMemory
)

class AgentState(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    ROUTING = "routing"
    COLLECTING_INFO = "collecting_info"
    OFFLINE = "offline"

    
class Memory:

        def _create_memory_system(self, session_id: str) -> Dict:
            """Create a combined memory system using different memory types."""

            buffer_window_memory = ConversationBufferWindowMemory(
                k=self.memory_config["window_size"],
                return_messages=True,
                output_key="output",
                input_key="input",
                memory_key="recent_messages"
            )

            token_buffer_memory = ConversationTokenBufferMemory(
                llm=self.llm,
                max_token_limit=self.memory_config["max_token_limit"],
                return_messages=True,
                output_key="output",
                input_key="input",
                memory_key="token_buffer"
            )

            summary_memory = ConversationSummaryMemory(
                llm=self.summary_llm,
                return_messages=True,
                output_key="output",
                input_key="input",
                memory_key="conversation_summary"
            )

            memory_system = {
                "buffer_window": buffer_window_memory,
                "token_buffer": token_buffer_memory,
                "summary": summary_memory,
                "message_count": 0,
                "session_id": session_id
            }

            return memory_system
        
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

            for memory_key in ["buffer_window", "token_buffer"]:
                memory = memory_system[memory_key]
                memory.save_context({"input": human_message},
                                    {"output": ai_message})

            memory_system["message_count"] += 1

            if memory_system["message_count"] % self.memory_config["summary_interval"] == 0:
                self._update_conversation_summary(session_id)

        def _update_conversation_summary(self, session_id: str) -> None:
            """Update the conversation summary based on token buffer memory."""
            memory_system = self._get_session_memory(session_id)

            token_buffer = memory_system["token_buffer"]
            buffer_messages = token_buffer.buffer

            if not buffer_messages:
                return

            chat_history = buffer_messages
            summary_input = {"chat_history": chat_history}

            try:
                summary = self._create_summary_chain().invoke(summary_input)

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
                token_buffer = memory_system["token_buffer"]
                token_buffer.max_token_limit = token_buffer.max_token_limit // 2

                self._update_conversation_summary(session_id)

                fallback_input = {
                    "input": input_text,
                    "conversation_summary": memory_system["summary"].buffer
                }

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

            memory_system = self._get_session_memory(session_id)

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

            if "memory_system" in session_memory:
                context.add_to_context(
                    "memory_system", session_memory["memory_system"])

