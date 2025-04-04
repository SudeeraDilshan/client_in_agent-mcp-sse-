from langchain.agents import tool


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


@tool
def offer(input: int) -> int:
    """Takes a number, doubles its value and adds 100."""
    return (input * 2) + 100


@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate the final price after applying a discount percentage.

    Args:
        price: The original price
        discount_percent: Discount percentage (0-100)

    Returns:
        The final price after discount
    """
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount percentage must be between 0 and 100")
    discount_amount = price * (discount_percent / 100)
    return price - discount_amount


inbuilt_tools = [magic_function, offer, calculate_discount]
