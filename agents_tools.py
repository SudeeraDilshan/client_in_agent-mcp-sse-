from typing import Dict, Any, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class MagicFunctionInput(BaseModel):
    """Input schema for the magic function."""
    input_text: str = Field(...,
                            description="The input text containing a number to add 10 to")


class MagicFunctionTool(BaseTool):
    """A tool that adds 10 to any input number."""

    name: str = "magic_function"
    description: str = "Use this tool when you need to add 10 to a number. Input should be a valid number (integer or decimal)."
    args_schema: Type[BaseModel] = MagicFunctionInput

    def _run(self, input_text: str) -> Dict[str, Any]:
        """
        Add 10 to the input number.

        Args:
            input_text (str): The input text containing a number

        Returns:
            dict: A dictionary containing the status and result
        """
        try:
            # Convert input to float to handle both integers and decimals
            number = float(input_text)
            result = number + 10

            return {
                "status": "success",
                "result": f"{number} + 10 = {result}"
            }
        except ValueError:
            return {
                "status": "error",
                "result": "Please provide a valid number"
            }

    async def _arun(self, input_text: str) -> Dict[str, Any]:
        """Async version of _run."""
        return self._run(input_text)


# List of inbuilt tools
inbuilt_tools = [MagicFunctionTool()]