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


async def get_pain_relief_advice(body_part: str) -> str:
    """Get advice for pain relief based on the body part."""
    advice_dict = {
        "head": "For headaches, rest in a quiet, dark room. Stay hydrated and consider over-the-counter pain relievers if appropriate. Apply a cold or warm compress to your forehead or neck. If headaches are severe or persistent, consult a doctor.",
        "neck": "Apply heat or cold to the painful area. Practice gentle stretching and improve your posture. Take over-the-counter pain relievers if needed. If pain is severe or includes numbness in arms, seek medical attention.",
        "back": "Rest your back but avoid complete bed rest. Apply ice for the first 48 hours, then switch to heat. Practice good posture and proper lifting techniques. Gentle stretches may help once acute pain subsides. Consult a doctor if pain is severe or persistent.",
        "shoulder": "Rest the affected shoulder and apply ice to reduce inflammation. Simple stretches and range-of-motion exercises may help. OTC pain relievers can reduce pain and swelling. See a doctor if pain persists or if there's significant weakness.",
        "arm": "Rest the affected arm and elevate it if possible. Apply ice to reduce swelling. Over-the-counter pain medication may help. If pain is severe or there's visible deformity, seek medical attention immediately.",
        "elbow": "Rest your elbow and avoid activities that cause pain. Apply ice and use compression if swelling is present. OTC anti-inflammatory medications may help. Consider using an elbow brace for support during recovery.",
        "wrist": "Rest your wrist and avoid activities that increase pain. Apply ice and consider using a wrist splint. Elevate your hand above heart level to reduce swelling. If pain persists or you experience numbness, see a doctor.",
        "hand": "Rest your hand and avoid gripping activities. Apply ice to reduce inflammation. Keep your hand elevated when possible. Gentle stretching may help. See a doctor if you have severe pain or inability to move fingers.",
        "hip": "Rest the affected hip and apply ice to reduce inflammation. Over-the-counter pain relievers may help. Avoid activities that worsen pain. If pain persists or walking becomes difficult, consult a doctor.",
        "knee": "Rest the knee and avoid weight-bearing activities that cause pain. Apply ice and elevate your leg to reduce swelling. Consider using a compression bandage and OTC pain relievers. See a doctor if you can't bear weight or if the knee locks.",
        "ankle": "Follow the RICE protocol: Rest, Ice, Compression, Elevation. Avoid putting weight on the affected ankle. OTC pain relievers may help reduce pain and swelling. If you can't walk or if there's severe swelling, seek medical attention.",
        "foot": "Rest your foot and avoid activities that cause pain. Apply ice to reduce inflammation. Consider supportive footwear or orthotics. If pain is severe or walking is difficult, consult a doctor.",
        "stomach": "Try sipping clear liquids and eating bland foods. Avoid spicy, fatty foods, caffeine, and alcohol. A heating pad on low setting may help. If pain is severe, sudden, or accompanied by fever or vomiting, seek medical attention immediately.",
        "chest": "Chest pain could be serious and may indicate a heart problem. If you're experiencing chest pain, especially with shortness of breath, sweating, or pain radiating to the arm or jaw, seek emergency medical help immediately.",
        "tooth": "Rinse with warm salt water and gently floss to remove any trapped food. Over-the-counter pain relievers may help temporarily. Apply a cold compress to the outside of your cheek. Schedule a dental appointment as soon as possible.",
        "ear": "Keep the ear dry and avoid inserting anything into the ear canal. Apply a warm compress to the outside of the ear. OTC pain relievers may help. If pain is severe, persistent, or accompanied by fever or hearing loss, consult a doctor.",
        "eye": "Rest your eyes and avoid straining them. Apply a cool compress for pain or swelling. Avoid touching or rubbing your eyes. If you experience severe pain, vision changes, or foreign object sensation, seek immediate medical attention.",
        "throat": "Stay hydrated and gargle with warm salt water. Suck on throat lozenges or ice chips. Use a humidifier to add moisture to the air. If sore throat is severe, lasts more than a week, or is accompanied by difficulty breathing, seek medical attention.",
        "joint": "Rest the affected joint and apply ice to reduce swelling. Avoid activities that cause pain. Over-the-counter anti-inflammatory medications may help. Consider gentle range-of-motion exercises once pain subsides. Consult a doctor if pain persists or worsens.",
        "muscle": "Rest the affected muscle and apply ice for the first 48 hours to reduce inflammation. After 48 hours, apply heat to increase blood flow. Over-the-counter pain relievers may help. Gentle stretching once pain subsides can aid recovery.",
    }

    body_part = body_part.lower()
    if body_part in advice_dict:
        return advice_dict[body_part]
    else:
        return f"For pain in your {body_part}, I recommend resting the area and applying ice to reduce inflammation. Over-the-counter pain relievers may provide temporary relief. If the pain is severe, persistent, or concerning, please consult a healthcare professional for proper diagnosis and treatment."


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

    elif name == "pain_relief":
        if "body_part" not in arguments:
            logger.error("Missing required argument 'body_part'")
            raise ValueError("Missing required argument 'body_part'")
        pain_advice = await get_pain_relief_advice(arguments["body_part"])
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
            name="pain_relief",
            description="Get advice on how to relieve pain in different body parts",
            inputSchema={
                "type": "object",
                "required": ["body_part"],
                "properties": {
                    "body_part": {
                        "type": "string",
                        "description": "The body part where you're experiencing pain (e.g., head, back, knee)",
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
