from langchain.agents import tool

@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

@tool
def offer(input: int) -> int:
    """Takes a number, doubles its value and adds 100."""
    return (input * 2) + 100

inbuilt_tools = [magic_function, offer]