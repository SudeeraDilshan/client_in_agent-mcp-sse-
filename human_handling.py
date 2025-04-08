from enum import Enum
from typing import Dict, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
from dialdeskai.src.agents.output import AgentOutput
from langdetect import detect


class AgentState(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    ROUTING = "routing"
    COLLECTING_INFO = "collecting_info"
    OFFLINE = "offline"


class HumanHandlingMixin:
    """Mixin class for human handling functionality in conversational agents."""

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

            routing_result = self._route_to_human(
                query, language, department, session_id, room_id)
            routing_info['human_connected'] = True
            response_message = routing_result.get(
                'message', f"Connecting you to {department} in {language}...")

            chat_history.extend(
                [HumanMessage(content=data), AIMessage(content=response_message)])

            self._update_memory(session_id, data, response_message)

            session_memory["chat_history"] = chat_history
            session_memory["routing_info"] = routing_info
            self._update_context(context, session_memory)

            return AgentOutput(result=response_message, metadata={"action": "human_handoff", "routing_result": routing_result})
        else:
            forward_message = "Your message has been forwarded to the human agent."
            chat_history.extend(
                [HumanMessage(content=data), AIMessage(content=forward_message)])

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

        if routing_info.get('language') and routing_info.get('department'):
            session_memory["state"] = AgentState.ROUTING
            self._update_context(context, session_memory)
            return self._handle_routing_state(context, data, session_memory)

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

        chat_history.extend([HumanMessage(content=data),
                            AIMessage(content=info_message)])
        self._update_memory(session_id, data, info_message)

        session_memory["chat_history"] = chat_history
        session_memory["routing_info"] = routing_info
        self._update_context(context, session_memory)

        return AgentOutput(result=info_message)

    def _analyze_conversation_for_department(self, session_id: str) -> Optional[str]:
        """Analyze the entire conversation history to determine the most likely department needed."""

        memory_system = self._get_session_memory(session_id)

        user_messages = []

        if "buffer_window" in memory_system:
            recent_messages = memory_system["buffer_window"].load_memory_variables(
                {}).get("recent_messages", [])
            for message in recent_messages:
                if isinstance(message, HumanMessage):
                    user_messages.append(message.content)

        if "token_buffer" in memory_system:
            token_messages = memory_system["token_buffer"].load_memory_variables(
                {}).get("token_buffer", [])
            for message in token_messages:
                if isinstance(message, HumanMessage):
                    user_messages.append(message.content)

        combined_text = " ".join(user_messages).lower()

        department_counts = {
            "sales": 0,
            "customer_service": 0,
            "technical_support": 0
        }

        for keyword, department in self.department_map.items():
            if keyword in combined_text:
                department_counts[department] += 1

        max_count = max(department_counts.values())
        if max_count > 0:
            for dept, count in department_counts.items():
                if count == max_count:
                    return dept

        return self._advanced_department_analysis(combined_text)

    def _advanced_department_analysis(self, text: str) -> Optional[str]:
        """Use more sophisticated analysis to determine department when keyword matching fails."""

        tech_patterns = [
            "not working", "broken", "error", "bug", "fix", "problem",
            "troubleshoot", "install", "update", "upgrade", "version",
            "crash", "slow", "login issue", "password reset", "access"
        ]

        sales_patterns = [
            "price", "cost", "discount", "purchase", "buy", "package",
            "subscription", "plan", "trial", "demo", "quote", "offer",
            "availability", "order", "delivery"
        ]

        cs_patterns = [
            "account", "bill", "charge", "refund", "cancel", "subscription",
            "policy", "complaint", "feedback", "service", "assistance",
            "help", "support", "information", "question"
        ]

        tech_count = sum(1 for pattern in tech_patterns if pattern in text)
        sales_count = sum(1 for pattern in sales_patterns if pattern in text)
        cs_count = sum(1 for pattern in cs_patterns if pattern in text)

        counts = {
            "technical_support": tech_count,
            "sales": sales_count,
            "customer_service": cs_count
        }

        max_count = max(counts.values())

        if max_count > 0:
            for dept, count in counts.items():
                if count == max_count:
                    return dept

        return "customer_service"

    def _handle_human_routing(self, context, data, session_memory):
        """Handle routing to a human agent with automatic department detection."""

        session_id = session_memory["routing_info"]["session_id"]
        room_id = session_memory["routing_info"]["room"]
        routing_info = session_memory["routing_info"]

        language = routing_info.get('language')
        if not language:
            detected_language = self._extract_language(data)
            if detected_language:
                language = detected_language
            else:
                language = "English"

        department = routing_info.get('department')
        if not department:
            department = self._extract_department(data)

            if not department:
                department = self._analyze_conversation_for_department(
                    session_id)

                if not department:
                    routing_info['language'] = language
                routing_info['query'] = data
                session_memory["state"] = AgentState.COLLECTING_INFO

                response_message = f"I'll connect you with a human agent. Which department do you need? Options: Sales, Customer Service, or Technical Support."

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

        routing_result = self._route_to_human(
            data, language, department, session_id, room_id
        )

        routing_info['language'] = language
        routing_info['department'] = department
        routing_info['human_connected'] = True
        session_memory["state"] = AgentState.ROUTING

        response_message = routing_result.get(
            'message', f"Connecting you to our {department} team who speaks {language}. A representative will assist you shortly."
        )

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

    def _extract_language(self, text: str) -> Optional[str]:
        """Extract language preference from text using improved detection."""
        text_lower = text.lower()

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
