import asyncio
from datetime import date
import logging
import os
from pathlib import Path

from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.models import (
    AsyncFunctionTool,
    AsyncToolSet,
    CodeInterpreterTool,
    FileSearchTool,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from sales_data import SalesData
from stream_event_handler import StreamEventHandler
from terminal_colors import TerminalColors as tc
from utilities import Utilities

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

TENTS_DATA_SHEET_FILE = "datasheet/contoso-tents-datasheet.pdf"
API_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
# BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MAX_COMPLETION_TOKENS = 4096
MAX_PROMPT_TOKENS = 10240
TEMPERATURE = 0.1
TOP_P = 0.1

toolset = AsyncToolSet()
sales_data = SalesData()
utilities = Utilities()
vector_store = None  # Will be set when tools are added

project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential()
)

functions = AsyncFunctionTool(
    {
        sales_data.async_fetch_sales_data_using_sqlite_query,
    }
)

INSTRUCTIONS_FILE = "instructions/instructions_function_calling.txt"
INSTRUCTIONS_FILE = "instructions/instructions_code_interpreter.txt"
INSTRUCTIONS_FILE = "instructions/instructions_file_search.txt"


async def add_agent_tools():
    """Add tools for the agent."""
    # Add the functions tool
    toolset.add(functions)

    # Add the code interpreter tool
    code_interpreter = CodeInterpreterTool()
    toolset.add(code_interpreter)

    # Add the tents data sheet to a new vector data store
    global vector_store
    vector_store = await utilities.create_vector_store(
        project_client,
        files=[TENTS_DATA_SHEET_FILE],
        vector_name_name="Contoso Product Information Vector Store",
    )
    print(f"Vector Store ID: {vector_store.id}")
    print(f"Vector Store Name: {vector_store.name}")
    print(f"Vector Store Status: {vector_store.status}")
    print(f"File Count: {vector_store.file_counts.total if vector_store.file_counts else 0}")
    
    # Add file search tool with vector store IDs
    file_search_tool = FileSearchTool(vector_store_ids=[vector_store.id])
    toolset.add(file_search_tool)
    print(f"Added FileSearchTool with vector store: {vector_store.id}")


async def initialize() -> tuple:
    """Initialize the agent with the sales data schema and instructions."""

    await add_agent_tools()

    await sales_data.connect()
    database_schema_string = await sales_data.get_database_info()

    try:
        env = os.getenv("ENVIRONMENT", "local")
        base_dir = Path(__file__).resolve().parent
        INSTRUCTIONS_FILE_PATH = base_dir.joinpath(INSTRUCTIONS_FILE)

        # Validate that the instructions file exists
        if not INSTRUCTIONS_FILE_PATH.is_file():
            raise FileNotFoundError(f"Instructions file not found: {INSTRUCTIONS_FILE_PATH}")

        # Ensure database connection is available
        if sales_data.conn is None:
            raise RuntimeError("Database connection unavailable. Check SQLite DB path and permissions.")

        with open(INSTRUCTIONS_FILE_PATH, "r", encoding="utf-8", errors="ignore") as file:
            instructions = file.read()

        # Replace the placeholder with the database schema string
        instructions = instructions.replace("{database_schema_string}", database_schema_string)
        instructions = instructions.replace("{current_date}", date.today().strftime("%Y-%m-%d"))

        print("Creating agent...")
        # Enable automatic function execution
        project_client.agents.enable_auto_function_calls(
            {sales_data.async_fetch_sales_data_using_sqlite_query}
        )
        
        # Create agent with toolset (which includes vector store via FileSearchTool)
        agent = await project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name="Contoso Sales AI Agent",
            instructions=instructions,
            toolset=toolset,
            temperature=TEMPERATURE,
            # headers={"x-ms-enable-preview": "true"},
        )
        print(f"Created agent, ID: {agent.id}")
        
        # Debug: Print agent's tools
        print(f"Agent tools: {[tool.type if hasattr(tool, 'type') else type(tool).__name__ for tool in (agent.tools if hasattr(agent, 'tools') else [])]}")
        if hasattr(agent, 'tool_resources') and agent.tool_resources:
            print(f"Tool resources: {agent.tool_resources}")

        print("Creating thread...")
        thread = await project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        return agent, thread

    except Exception as e:
        logger.error("An error occurred initializing the agent: %s", str(e))
        logger.error("Please ensure the instructions file exists and the database is accessible.")
        raise


async def cleanup(agent, thread) -> None:
    """Cleanup the resources."""
    # Note: Thread and agent cleanup - check if delete methods exist in your SDK version
    try:
        await project_client.agents.threads.delete(thread.id)
    except AttributeError:
        pass  # Delete may not be available in all SDK versions
    try:
        await project_client.agents.delete(agent.id)
    except AttributeError:
        pass  # Delete may not be available in all SDK versions
    await sales_data.close()
    # Close the AIProjectClient to avoid unclosed session warnings
    await project_client.close()


async def post_message(thread_id: str, content: str, agent, thread) -> None:
    """Post a message to the Azure AI Agent Service."""
    try:
        await project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
        )

        # Stream the run directly using the event handler
        stream = await project_client.agents.runs.stream(
            thread_id=thread.id,
            agent_id=agent.id,
            event_handler=StreamEventHandler(functions=functions, project_client=project_client, utilities=utilities),
        )
        
        async with stream as s:
            await s.until_done()
    except Exception as e:
        utilities.log_msg_purple(f"An error occurred posting the message: {str(e)}")


async def main() -> None:
    """
    Main function to run the agent.
    Example questions: Sales by region, top-selling products, total shipping costs by region, show as a pie chart.
    """
    try:
        agent, thread = await initialize()
    except Exception as e:
        logger.error("Initialization failed: %s", str(e))
        print("Initialization failed. Check logs and configuration, then try again.")
        return

    try:
        while True:
            # Get user input prompt in the terminal using a pretty shade of green
            print("\n")
            prompt = input(f"{tc.GREEN}Enter your query (type exit to finish): {tc.RESET}")
            if prompt.lower() == "exit":
                break
            if not prompt:
                continue
            
            await post_message(agent=agent, thread_id=thread.id, content=prompt, thread=thread)
    finally:
        await cleanup(agent, thread)


if __name__ == "__main__":
    print("Starting async program...")
    asyncio.run(main())
    print("Program finished.")
