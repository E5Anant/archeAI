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

def web_opener(url:str):
    """Opens the given URL in the default web browser.

    Args:
        url (str): The URL to open in the web browser.
    """
    import webbrowser
    webbrowser.open(url)

llm_instance = Gemini()

# Define the tools using the OwnTool class
write_tool = Tool(
    func=write_to_file,
    description="Writes the given content to a file to the given filename in dir",
    returns_value=False,
    llm = llm_instance,
    verbose=True
)
website_opener = Tool(
    func=web_opener,
    description="Opens the given URL in the default web browser.",
    returns_value=False,
    llm=llm_instance,
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
        identity="ChatBot",
        tools=[write_tool, list_tool, time_tool, web_tool, website_opener],
        description="a powerfull ai agent",
        memory=False,
        objective = "get information about wormholes in the universe and save the article in wormholes.md",
        memory_dir="MEMORIES",
        verbose=True,
        max_memory_responses=1
                )


if __name__ == "__main__":
    while True:
    # Create the agent with multiple tools
        TaskForce([Chatbot], objective=input(">>> ")).rollout()
    # re
    # sult = print(Chatbot.rollout())