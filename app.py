from archeai.llms import Gemini
from archeai import Agent, Tool
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

Chatbot = Agent(
        llm=llm_instance,
        name="ChatBot",
        tools=[write_tool, list_tool, time_tool, web_tool],
        description="a powerfull ai agent",
        expected_output="",
        task="",
        memory=False,
        memory_dir="MEMORIES",
        verbose=True,
        max_memory_responses=1
                )


if __name__ == "__main__":
    # Create the agent with multiple tools
    Chatbot.task = "Get the current time and what should we do at this time and write the article in a file time.txt."
    Chatbot.rollout()
    # re
    # sult = print(Chatbot.rollout())