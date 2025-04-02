from langchain.agents import tool

@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

inbuilt_tools = [magic_function]