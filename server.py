from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from mcp.server import Server
import mcp.types as types
import logging
import httpx

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="server.log", filemode="w")
logger = logging.getLogger(__name__)

app = Server("example-server")
sse = SseServerTransport("/messages/")

ToolCallResult = list[types.TextContent | types.ImageContent | types.EmbeddedResource]


async def get_mood_advice(mood: str) -> str:
    """Get advice based on the user's current mood."""
    advice_dict = {
        "happy": "Great to hear you're feeling happy! Use this positive energy to tackle challenges or help others.",
        "sad": "I'm sorry you're feeling sad. Consider talking to someone you trust, or doing something you enjoy to lift your spirits.",
        "anxious": "Take a few deep breaths and try to focus on the present moment. Consider what specific things are causing your anxiety.",
        "stressed": "Try to step back and prioritize tasks. Consider taking short breaks and practicing mindfulness or relaxation techniques.",
        "bored": "This could be a good opportunity to learn something new or reconnect with an old hobby you enjoy.",
        "tired": "Listen to your body. If possible, take a short rest or consider if you need to adjust your sleep schedule.",
        "angry": "Try to step back from the situation. Consider if expressing your feelings calmly could help resolve what's bothering you.",
        "excited": "Channel that excitement into something productive. Set clear goals to make the most of this energy.",
        "confused": "Break down what's confusing you into smaller parts. Sometimes writing things out can help clarify your thoughts.",
    }

    mood = mood.lower()
    if mood in advice_dict:
        return advice_dict[mood]
    else:
        return f"I don't have specific advice for the mood '{mood}', but remember that all emotions are valid and temporary. Consider what might help you feel more balanced."


async def get_pain_advice(pain: str) -> str:
    """Get advice based on the user's described pain."""
    advice_dict = {
        "headache": "For headaches, rest in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers if appropriate. If headaches are severe or persistent, consult a doctor.",
        "back pain": "For back pain, maintain good posture and avoid heavy lifting. Apply ice for acute pain and heat for chronic pain. Gentle stretching may help, but stop if pain increases. Consult a doctor if pain is severe or persistent.",
        "muscle pain": "Rest the affected muscle and apply ice to reduce inflammation. Over-the-counter pain relievers may help. Consider gentle stretching once pain subsides. If pain is severe or lasts more than a few days, consult a doctor.",
        "joint pain": "Rest the affected joint and apply ice to reduce swelling. Avoid activities that cause pain. Over-the-counter anti-inflammatory medications may help. Consult a doctor if pain persists or is accompanied by swelling.",
        "stomach pain": "For mild stomach pain, stay hydrated and eat bland foods. Avoid spicy, fatty foods and alcohol. If pain is severe, sudden, or accompanied by fever or vomiting, seek medical attention immediately.",
        "chest pain": "Chest pain could be serious. If you're experiencing chest pain, especially with shortness of breath, sweating, or pain radiating to the arm or jaw, seek emergency medical help immediately as it could be a heart attack.",
        "tooth pain": "Rinse with warm salt water and gently floss to remove any trapped food. Over-the-counter pain relievers may help temporarily. Schedule a dental appointment as soon as possible, as tooth pain often indicates a problem requiring treatment.",
        "sore throat": "Stay hydrated and gargle with warm salt water. Throat lozenges or over-the-counter pain relievers may help. If sore throat is severe, lasts more than a week, or is accompanied by difficulty breathing, seek medical attention.",
        "ear pain": "Keep the ear dry and avoid inserting anything into the ear canal. Over-the-counter pain relievers may help. If pain is severe, persistent, or accompanied by fever or hearing loss, consult a doctor.",
    }

    pain = pain.lower()
    if pain in advice_dict:
        return advice_dict[pain]
    else:
        return f"For pain described as '{pain}', I recommend monitoring your symptoms and consulting a healthcare professional if the pain is severe, persistent, or concerning. Remember that proper medical advice should always be sought for health concerns."


async def fetch_website(
    url: str,
) -> ToolCallResult:
    logger.debug(f"Fetching website content from URL: {url}")
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        logger.debug(f"Received response with status code: {response.status_code}")
        return [types.TextContent(type="text", text=response.text)]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> ToolCallResult:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    if name == "mood_advice":
        if "mood" not in arguments:
            logger.error("Missing required argument 'mood'")
            raise ValueError("Missing required argument 'mood'")
        mood_result = await get_mood_advice(arguments["mood"])
        return [types.TextContent(type="text", text=mood_result)]

    elif name == "pain_advice":
        if "pain" not in arguments:
            logger.error("Missing required argument 'pain'")
            raise ValueError("Missing required argument 'pain'")
        pain_advice = await get_pain_advice(arguments["pain"])
        return [types.TextContent(type="text", text=pain_advice)]

    elif name == "fetch":
        if "url" not in arguments:
            logger.error("Missing required argument 'url'")
            raise ValueError("Missing required argument 'url'")
        return await fetch_website(arguments["url"])


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("Listing available tools.")
    return [
        types.Tool(
            name="mood_advice",
            description="Get advice based on your current mood",
            inputSchema={
                "type": "object",
                "required": ["mood"],
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "Your current mood or emotional state",
                    }
                },
            },
        ),
        types.Tool(
            name="pain_advice",
            description="Get advice on how to be careful and protected when experiencing pain",
            inputSchema={
                "type": "object",
                "required": ["pain"],
                "properties": {
                    "pain": {
                        "type": "string",
                        "description": "The type of pain you are experiencing",
                    }
                },
            },
        ),
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
    ]


async def handle_sse(request):
    try:
        logger.debug(f"Handling new SSE connection from: {request.client}")
        logger.info("SSE connection attempt received")
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            logger.info("SSE connection established successfully")
            await app.run(streams[0], streams[1], app.create_initialization_options())
    except Exception as e:
        logger.error(f"Error in SSE connection: {e}", exc_info=True)


from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
]

starlette_app = Starlette(
    debug=True,
    middleware=middleware,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000")
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000, log_level="debug")
