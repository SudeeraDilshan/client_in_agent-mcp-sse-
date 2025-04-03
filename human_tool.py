import json
import os
from typing import ClassVar, List, Type, Dict
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
load_dotenv()

class HumanRoutingInput(BaseModel):
    """Input schema for the human routing tool."""
    query: str = Field(..., description="The user's query that needs to be routed")
    language: str = Field(..., description="The preferred language for communication")
    department: str = Field(None, description="The department to route to(Optional))")
    sessionId: str = Field(..., description="Unique session ID from external sources")
    session: str = Field(..., description="Unique session ID from external sources")
    room: str = Field(..., description="Unique room ID from external sources")

class HumanRoutingTool(BaseTool):
    """A tool to route queries to human agents based on language and department preferences.
     _ask_user_for_department to get the department
    You can use this tool to execute if you doesn't know the exact answer or you don't know the exact query to execute and
    such as quering account balance
    """
    
    name: str = "human_routing_tool"
    description: str = "Routes queries to human agents based on language and department preferences _ask_user_for_department to get the department and You can use this tool to execute if you doesn't know the exact answer or you don't know the exact query to execute.such as quering account balance"
    args_schema: Type[BaseModel] = HumanRoutingInput 
    ards_api_url: str = os.getenv("ARDS_API_URL")


    available_languages: List[str] = ["en", "es", "fr", "de", "zh"]  
    available_departments: List[str] = ["sales", "customer_service", "technical_support"] 

    LANGUAGE_MAP: ClassVar[Dict[str, str]] = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh"
    }

    def __init__(self, ards_api_url: str = None):
        """Initialize the routing tool with API configuration."""
        super().__init__()
        if ards_api_url:
            # self.agent_executor = agent_executor,
            self.ards_api_url = ards_api_url.rstrip('/') 
            

    def _ask_user_for_department(self) -> str:
        """Ask the user to select a department dynamically."""

        print("\nAvailable Departments:", ", ".join(self.available_departments))
        while True:
            department = input("Please select a department to route the call: ").strip().lower()
            if department in self.available_departments:
                return department
            print("Invalid department. Please choose from:", ", ".join(self.available_departments))

    def _validate_inputs(self, language: str, department: str) -> tuple[bool, str]:
        """Validate the input language and department."""
        
        normalized_language = language.strip().title()
    
        language_code = self.LANGUAGE_MAP.get(normalized_language, language.lower())
        
        if language_code not in self.available_languages:
            return False, f"Language '{language}' not supported. Available: {', '.join(self.available_languages)}"
        
        if department not in self.available_departments:
            return False, f"Department '{department}' not found. Available: {', '.join(self.available_departments)}"
        
        return True, ""

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare HTTP headers for API requests."""
        return {'Content-Type': 'application/json'} 
    

    def _call_routing_api(self, query: str, department: str, sessionId: str, language: str, session: str, room: str) -> Dict:
        """Make the API call to route the details."""

        typehuman = 'human'
        
        endpoint = self.ards_api_url
        payload = {
                   "id": sessionId, 
                   "AgentType": typehuman, 
                   "department": department, 
                   "Session": session,
                   "Room": room,
                   "languages": [language], 
                   }
        json_payload = json.dumps(payload,ensure_ascii=False)  

        try:
            response = requests.post(endpoint, headers=self._prepare_headers(), json=payload)
            response.raise_for_status()


            response_data = response.json()


            if response_data.get("success") == True:

                agent_name = response_data.get("agent", {}).get("id", "an agent")


                message = f"Your call has been routed to the {department} department. The available agent, {agent_name}, will handle your call shortly."
                
            else:
                message = "We are currently unable to route your call. Please try again later."


            return message

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API call failed: {str(e)}")
    
    def _run(self, query: str, language: str, department: str = None, sessionId: str = None, message: str = None,session: str = None, room: str = None) -> Dict:
        """Execute the routing logic.Ask for the department if it not available"""
        
        if not department:
            department = self._ask_user_for_department()

        is_valid, error_message = self._validate_inputs(language, department)
        if not is_valid:
            raise ValueError(error_message)

        message  = self._call_routing_api(query, department,sessionId, language,session,room )

        return {
            'status': 'success',
            'message': message
        }

    async def _arun(self, query: str, language: str, department: str = None, sessionId: str = None, session: str = None, room: str = None) -> Dict:
        """Async version of _run."""
        return self._run(query, language, department, sessionId, session, room)