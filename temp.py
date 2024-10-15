from archeai.llms import Gemini
from archeai import Agent, Tool, TaskForce
from archeai.tools import get_current_time, web_search
import os

def list_dir():
    """Returns a list of items in the current working directory."""

    try:
        items = os.listdir()
        return items
    except OSError as e:
        print(f"Error listing directory: {e}")
        return []
def write_to_file(filename:str, content:str):
    """Writes the given content to a file with the specified filename.

    Args:
        filename (str): The name of the file to write to.
        content (str): The content to write to the file.
    """

    try:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Successfully wrote to '{filename}'")
    except OSError as e:
        print(f"Error writing to file: {e}")

def gcd(a:int, b:int):
    """
    Calculate the Greatest Common Divisor (GCD) of two numbers using the Euclidean algorithm.

    Parameters:
    a (int): The first number.
    b (int): The second number.

    Returns:
    int: The GCD of the two numbers.
    """
    while b:
        a, b = b, a % b
    return a+b

llm_instance = Gemini()

# Define the tools using the OwnTool class
write_tool = Tool(
    func=write_to_file,
    description="Writes the given content to a file to the given filename in dir",
    returns_value=False,
    llm = llm_instance,
    verbose=True
)

list_tool = Tool(
    func=list_dir,
    description="Provides the list of files and folders in current working dir.",
    returns_value=True,
    llm = llm_instance,
    verbose=True
)

time_tool = Tool(
    func=get_current_time,
    description="Provides the current time.",
    returns_value=True,
    llm = llm_instance,
    verbose=True
)

web_tool = Tool(
    func=web_search,
    description="Provides web search result on the given query.",
    returns_value=True,
    llm = llm_instance,
    verbose=True
)

# Initialize the language model instance

content_saver = Agent(
        llm=llm_instance,
        identity="content_saver",
        tools=[write_tool],
        verbose=True,
        memory_dir="MEMORIES",
        memory=False,
        description="A content saver which can save any content in markdown format to a given file.",
        expected_output="In markdown format",
)

researcher = Agent(
        llm=llm_instance,
        identity="researcher",
        tools=[time_tool, web_tool],
        description="A researcher having access to web and can get info about any topic.",
        expected_output="A summary of the research",
        memory=False,
        memory_dir="MEMORIES",
        verbose=True,
                )

writer = Agent(
        llm=llm_instance,
        identity="writer",
        tools=[],
        description="A writer which can write on any topic with information.",
        expected_output="In markdown format",
        memory=False, 
        memory_dir="MEMORIES",
        verbose=True,
                )

team = TaskForce(agents=[researcher, writer, content_saver], objective="get information about wormholes in the universe and save the full informative article in wormholes.md",)

# team.assign_task(task="get information about wormholes in the universe and save the article in wormholes.md")

team.rollout()