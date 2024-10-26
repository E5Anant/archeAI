# Welcome to ArcheAI! 🚀

Building AI agents should feel like assembling a dream team, not wrestling with complex code. That's where **ArcheAI** comes in. ✨

**ArcheAI is a lightweight Python framework designed to make AI agent development intuitive, flexible, and downright fun!** 🎉

You might be thinking: *"Another AI agent framework? What makes ArcheAI different?"* 🤔

## Why ArcheAI? 💡

Let's face it, building sophisticated AI agents can feel overwhelming. ArcheAI is different because it's built on these core principles:

-   **Simplicity First:** 🧘 ArcheAI strips away unnecessary complexity, giving you a clean and elegant API that gets straight to the point. Focus on what matters -- building awesome agents!
-   **Unleash Your Creativity:** 🎨 We believe in giving you the tools, not dictating the path. ArcheAI's modular design empowers you to experiment with different LLMs (like Gemini, Groq, Cohere, OpenAI, and Anthropic), craft custom tools, and define your agent's unique personality.
-   **Power in Collaboration:** 🤝 Great things happen when minds work together! ArcheAI's **TaskForce** feature makes it a breeze to orchestrate multiple agents, allowing them to collaborate seamlessly on complex tasks. 
-   **Built for Exploration:** 🔭 The world of AI is constantly evolving. ArcheAI embraces this by providing a flexible foundation that's ready to adapt to new LLMs, tools, and possibilities. The only limit is your imagination!

## Installation ⚙️

Get started with ArcheAI in just a few steps:

```bash
pip install archeai 
```

## Core Concepts 📚

### The `Agent` Class 👤

The `Agent` class is the heart of ArcheAI. It represents an individual AI agent within your system.

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `llm` | An instance of the LLM (Large Language Model) class that the agent will use for language processing and decision-making. |
| `tools` | A list of `Tool` objects that define the actions the agent can perform. |
| `identity` | A string representing the agent's name or identifier. |
| `description` | A brief description of the agent's role or purpose. |
| `expected_output` | A string describing the expected format or style of the agent's responses. |
| `objective` | The current task or goal that the agent is trying to achieve. |
| `memory` | A boolean value indicating whether the agent should use memory (to retain context from previous interactions). Defaults to `True`. |
| `memory_dir` | The directory where the agent's memory files will be stored. Defaults to `memories`. |
| `max_chat_responses` | The maximum number of previous conversation turns to store in memory. Defaults to `12`. |
| `max_summary_entries` | The maximum number of summary entries to store in memory. Defaults to `3`. |
| `max_iterations` | The maximum number of iterations the agent will attempt to generate a valid response. Defaults to `3`. |
| `check_response_validity` | A boolean value indicating whether the agent should check the validity of the response before it is returned. Defaults to `True`. |
| `verbose` | A boolean value indicating whether the agent should print verbose output during execution. Defaults to `False`. |

#### Methods

- `add_tool(tool)`           Adds a `Tool` object to the agent's `tools` list.
- `remove_tool(tool_name)`   Removes a tool from the agent's `tools` list by its name.
- `rollout()`                Executes the agent's main workflow, including processing the `objective`, using tools, and generating a response.

### The `Tool` Class 🧰

The `Tool` class represents an action or capability that an agent can perform. Tools can be simple functions, complex operations, or even integrations with external services.

ArcheAI's `Tool` class is designed to make it incredibly easy to define and use tools within your agents. You don't need to write complex wrappers or adapters. Simply provide the core logic of your tool as a Python function, and ArcheAI will handle the rest!

**Example:**

```python
from archeai import Tool

def get_weather(city: str):
    """Fetches the current weather for a given city.""" 
    # ... (Implementation to fetch weather data) ... 
    return weather_data 

weather_tool = Tool(func=get_weather, 
                   description="Gets the current weather for a specified city.",
                    params={'city': {'description': 'The city for the weather to find.', 'type': 'str', 'default': 'unknown'}})
```

In this example, we define a simple tool called `weather_tool`. The tool uses the `get_weather` function to fetch weather data for a given city. The `description` parameter provides a concise explanation of what the tool does, which is helpful for both you and the agent to understand its purpose.

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `func` | The Python function that defines the tool's action. |
| `name` | The name of the tool (automatically derived from the function name). |
| `description` | A brief description of what the tool does. This is used by the agent to understand the tool's purpose. |
| `returns_value` | A boolean indicating whether the tool returns a value that can be used by other tools or included in the response. Defaults to `True`. |
| `instance` | Optional instance of a class if the tool is a bound method. |
| `llm` | Optional LLM object for more advanced tool interactions (e.g., using the LLM to help determine tool parameters). |
| `verbose` | A boolean indicating whether the tool should print verbose output during execution. Defaults to `False`. |
| `params` | An Optional dictionary containing information about the tool's parameters (automatically extracted if not provided). |

### The `TaskForce` Class 👥

The `TaskForce` class lets you manage a group of `Agent` objects, enabling collaboration and complex workflow orchestration.

#### Attributes

| Attribute | Description |
|-----------|-------------|
| `agents` | A list of `Agent` objects that belong to the task force. |
| `objective` | A overall goal or task that the task force is trying to achieve. |

#### Methods

- `rollout()`   Starts the task force's execution. This method intelligently assigns the `objective` to the most suitable agent and manages the workflow.

## Basic Example: Building Your First Team 🧑‍🤝‍🧑

Let's bring it all together with a simple example:

```python
from archeai import Agent, Tool, TaskForce
from archeai.llms import Gemini 

# Initialize your LLM
llm = Gemini()

# Define a greeting tool
def say_hello(name: str):
  return f"Hello there, {name}! 👋" 

def calculate(equation: str):
  return eval(equation)

hello_tool = Tool(func=say_hello, 
                   description="Greets the user by name.",
                    params={'name': {'description': 'The name to greet.', 'type': 'str', 'default': 'unknown'}})

calculate_tool = Tool(func=calculate, 
                   description="Evaluates an equation.",
                    params={'equation': {'description': 'The equation to evaluate.', 'type': 'str', 'default': 'unknown'}})

# Create an agent
greeter = Agent(llm=llm, 
                tools=[hello_tool], 
                identity="Friendly Greeter",
                memory=False,
                verbose=True)

math_magician = Agent(llm=llm, 
                tools=[calculate_tool], 
                identity="Math Magician",
                memory=False,
                verbose=True)

# Assemble your task force!
my_taskforce = TaskForce(agents=[greeter, math_magician], 
                         objective="Hi I am Mervin greet me, can you solve `3-4*2*5/4/2.1*6` for me and give a explanation.") 

# Start the interaction
response = my_taskforce.rollout() 
```

This basic example shows how to create a simple agent with a tool and then use a `TaskForce` to manage its execution.

## Make sure to star this repo if you liked it. 

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).