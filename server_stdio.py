from mcp.server.fastmcp import FastMCP 
import logging

mcp =FastMCP("mcp-server") 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting MCP server...")
logging.info("MCP server started successfully.")
logging.info("Listening for incoming requests...")


@mcp.tool()
def get_red_value(b:int) -> int:
    return b+5

@mcp.tool()
def get_blue_value(b:int) -> int:
    return b+10


if __name__ =="__main__":
    mcp.run(transport="stdio")