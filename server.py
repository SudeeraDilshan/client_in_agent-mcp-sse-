from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from mcp.server import Server
import mcp.types as types
import logging
import aiohttp

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="server.log", filemode="w")
logger = logging.getLogger(__name__)

app = Server("example-server")
sse = SseServerTransport("/messages/")

async def fetch_website(url: str) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Fetch content from a website URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return [types.TextContent(type="text", text=f"Content from {url}:\n{content[:1000]}...")]
                else:
                    return [types.TextContent(type="text", text=f"Failed to fetch {url}. Status code: {response.status}")]
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}", exc_info=True)
        return [types.TextContent(type="text", text=f"Error fetching {url}: {str(e)}")]

async def get_synonyms(word: str) -> list[str]:
    """Get synonyms for a given word using a predefined dictionary."""
    # Simple dictionary of synonyms for common words
    synonym_dict = {
        "happy": ["joyful", "cheerful", "content", "pleased", "delighted"],
        "sad": ["unhappy", "sorrowful", "dejected", "gloomy", "depressed"],
        "good": ["excellent", "fine", "great", "positive", "satisfactory"],
        "bad": ["poor", "awful", "terrible", "negative", "substandard"],
        "big": ["large", "huge", "enormous", "massive", "gigantic"],
        "small": ["tiny", "little", "miniature", "compact", "microscopic"],
        # Add more words as needed
    }
    
    word = word.lower()
    if word in synonym_dict:
        # Return only 3 synonyms as requested
        return synonym_dict[word][:3]
    else:
        return ["No synonyms found"]

async def get_description(subject: str) -> str:
    """Get a short description about a given object or concept."""
    description_dict = {
        "computer": "An electronic device for storing and processing data according to instructions given to it.",
        "internet": "A global computer network providing information and communication facilities.",
        "cloud computing": "The practice of using remote servers on the internet to store, manage, and process data.",
        "artificial intelligence": "The simulation of human intelligence processes by machines, especially computer systems.",
        "smartphone": "A mobile phone that performs many of the functions of a computer.",
        "blockchain": "A system in which a record of transactions is maintained across several computers linked in a peer-to-peer network.",
        "robot": "A machine capable of carrying out a complex series of actions automatically.",
        "virtual reality": "Computer-generated simulation of a three-dimensional environment that can be interacted with.",
        "neural network": "A computing system inspired by the biological neural networks that constitute animal brains.",
        "database": "An organized collection of data stored and accessed electronically.",
    }
    
    subject = subject.lower()
    if subject in description_dict:
        return description_dict[subject]
    else:
        return f"Sorry, I don't have information about '{subject}'."

@app.call_tool()
async def fetch_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.debug(f"Tool called: {name} with arguments: {arguments}")
    if name != "fetch":
        logger.error(f"Unknown tool: {name}")
        raise ValueError(f"Unknown tool: {name}")
    if "url" not in arguments:
        logger.error("Missing required argument 'url'")
        raise ValueError("Missing required argument 'url'")
    return await fetch_website(arguments["url"])

@app.call_tool()
async def get_synonyms_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    if name != "get_synonyms":
        logger.error(f"Unknown tool: {name}")
        raise ValueError(f"Unknown tool: {name}")
    if "word" not in arguments:
        logger.error("Missing required argument 'word'")
        raise ValueError("Missing required argument 'word'")
    
    word = arguments["word"]
    logger.info(f"Finding synonyms for: {word}")
    synonyms = await get_synonyms(word)
    return [types.TextContent(type="text", text=f"Synonyms for '{word}': {', '.join(synonyms)}")]

@app.call_tool()
async def describe_object(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    if name != "describe":
        logger.error(f"Unknown tool: {name}")
        raise ValueError(f"Unknown tool: {name}")
    if "subject" not in arguments:
        logger.error("Missing required argument 'subject'")
        raise ValueError("Missing required argument 'subject'")
    
    subject = arguments["subject"]
    logger.info(f"Getting description for: {subject}")
    description = await get_description(subject)
    return [types.TextContent(type="text", text=description)]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("Listing available tools.")
    return [
        types.Tool(
            name="fetch",
            description="Fetches a website and returns its content",
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch",
                    }
                },
            },
        ),
        types.Tool(
            name="get_synonyms",
            description="Get 3 synonyms for a given word",
            inputSchema={
                "type": "object",
                "required": ["word"],
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word to find synonyms for",
                    }
                },
            },
        ),
        types.Tool(
            name="describe",
            description="Get a short description about an object or concept",
            inputSchema={
                "type": "object",
                "required": ["subject"],
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The object or concept to describe",
                    }
                },
            },
        ),
    ]
        

async def handle_sse(request):
    try:
        logger.debug(f"Handling new SSE connection from: {request.client}")
        logger.info("SSE connection attempt received")
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            logger.info("SSE connection established successfully")
            await app.run(streams[0], streams[1], app.create_initialization_options())
    except Exception as e:
        logger.error(f"Error in SSE connection: {e}", exc_info=True)
        # We still need to handle the error properly here

async def handle_messages(request):
    try:
        logger.info("Message received")
        await sse.handle_post_message(request.scope, request.receive, request._send)
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"])
]

starlette_app = Starlette(
    debug=True,
    middleware=middleware,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000")
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000, log_level="debug")