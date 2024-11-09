

<div align="center">

# Welcome to ArcheAI! 🚀

*Building AI agents should feel like assembling a dream team, not wrestling with complex code.*

[![PyPI version](https://badge.fury.io/py/archeai.svg)](https://badge.fury.io/py/archeai)
[![GitHub stars](https://img.shields.io/github/stars/E5Anant/archeAI.svg?style=social&label=Star)](https://github.com/E5Anant/archeAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://github.com/E5Anant/archeAI#readme) | [PyPI Package](https://pypi.org/project/archeai/) | [GitHub Repository](https://github.com/E5Anant/archeAI)

</div>

---

## 🌟 Why ArcheAI?

<div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px;">

ArcheAI is a lightweight Python framework designed to make AI agent development intuitive, flexible, and downright fun! 🎉

> "Another AI agent framework? What makes ArcheAI different?" 🤔

Let's break it down:

1. **Simplicity First** 🧘
   - Strip away unnecessary complexity
   - Clean and elegant API
2. **Unleash Your Creativity** 🎨
   - Modular design for experimentation
   - Support for various LLMs (Gemini, Groq, Cohere, OpenAI, Anthropic)
3. **Power in Collaboration** 🤝
   - TaskForce feature for seamless agent orchestration
4. **Built for Exploration** 🔭
   - Flexible foundation ready to adapt to new technologies

</div>

---

## ⚙️ Installation

Get started with ArcheAI in just one line:

```bash
pip install archeai
```

---

## 📚 Core Concepts

<details open>
<summary><h3>👤 The Agent Class</h3></summary>

The Agent class is the heart of ArcheAI. It represents an individual AI agent within your system.

#### Key Attributes

| Attribute | Description |
|-----------|-------------|
| `llm` | An instance of the LLM (Large Language Model) class that the agent will use for language processing and decision-making. |
| `tools` | A list of `Tool` objects that define the actions the agent can perform. |
| `identity` | A string representing the agent's name or identifier. |
| `description` | A brief description of the agent's role or purpose. |
| `expected_output` | A string describing the expected format or style of the agent's responses. |
| `objective` | The current task or goal that the agent is trying to achieve. |
| `memory` | A boolean value indicating whether the agent should use memory (to retain context from previous interactions). Defaults to `True`. |
| `ask_user` | A boolean value indicating whether the agent should ask the user for input when generating responses. Defaults to `True`. |
| `memory_dir` | The directory where the agent's memory files will be stored. Defaults to `memories`. |
| `max_chat_responses` | The maximum number of previous conversation turns to store in memory. Defaults to `12`. |
| `max_summary_entries` | The maximum number of summary entries to store in memory. Defaults to `3`. |
| `max_iterations` | The maximum number of iterations the agent will attempt to generate a valid response. Defaults to `3`. |
| `check_response_validity` | A boolean value indicating whether the agent should check the validity of the response before it is returned. Defaults to `False`. |
| `allow_full_delegation` | Whether the agent should allow full delegation switching to `True` would give all the previous responses from different agents. Switching to False would only allow the last response, Defaults to `False` |
| `output_file` | The name of the file where the agent's responses will be saved. Defaults to `None`. |
| `verbose` | A boolean value indicating whether the agent should print verbose output during execution. Defaults to `False`. |


#### Methods

- `add_tool(tool)`: Add a new tool to the agent
- `remove_tool(tool_name)`: Remove a tool by name
- `rollout()`: Execute the agent's main workflow

</details>

<details>
<summary><h3>🧰 The Tool Class</h3></summary>

Tools are actions or capabilities that an agent can perform.

```python
from archeai import Tool

def get_weather(city: str):
    """Fetches the current weather for a given city."""
    # Implementation...
    return weather_data

weather_tool = Tool(
    func=get_weather,
    description="Gets the current weather for a specified city.",
    params={'city': {'description': 'The city to check weather for', 'type': 'str', 'default': 'unknown'}}
)
```

#### Key Attributes

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

</details>

<details>
<summary><h3>👥 The TaskForce Class</h3></summary>

Manage a group of Agent objects for collaboration and complex workflows.

#### Key Attributes

- `agents`: List of Agent objects in the task force
- `objective`: Overall goal or task for the task force
- `mindmap`: Overall plan or mind map (auto-generated if not provided)

#### Key Methods

- `rollout()`: Starts the task force's execution

<details>
<summary>Workflow Diagram</summary>

![Workflow](https://raw.githubusercontent.com/E5Anant/archeAI/main/assets/WorkFlow.png)

</details>

</details>

---

## 🧑‍🤝‍🧑 Basic Example: Building Your First Team

<div style="background-color: #e6f7ff; padding: 15px; border-radius: 5px;">

```python
from archeai import Agent, Tool, TaskForce
from archeai.llms import Gemini

# Initialize your LLM
llm = Gemini()

# Define tools
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

# Create agents
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
                         objective="Hi I am Mervin greet me, can you solve 3-4*2*5/4/2.1*6 for me and give a explanation.") 

# Start the interaction
response = my_taskforce.rollout()
print(response)
```

This example demonstrates creating agents with tools and using a TaskForce to manage execution.

</div>

---

## 🧐 Important Questions

<details>
<summary><strong>What is MindMap?</strong></summary>

A mind map is a visual representation of the task force's workflow and goals. It's automatically generated if not provided, helping to organize and structure the collaboration between agents.

</details>

<details>
<summary><strong>What does allow_full_delegation mean?</strong></summary>

The `allow_full_delegation` parameter controls how much information is shared between agents:

- When `False` (default): Only the last response is shared
- When `True`: All previous responses from different agents are shared

This allows for more comprehensive or limited collaboration depending on your needs.

</details>

---

## 🚀 Advanced Features

- **Multi-LLM Support**: Seamlessly switch between different language models
- **Custom Tool Creation**: Easily create and integrate your own tools
- **Memory Management**: Fine-tune agent memory for context retention
- **Response Validation**: Ensure output quality with built-in validation

---

## 📈 Performance and Scalability

ArcheAI is designed for efficiency:

- Lightweight core for minimal overhead
- Asynchronous capabilities for improved performance
- Scalable architecture for complex agent networks

---

<div align="center">

## 🤝 Contributing

We welcome contributions! Please feel free to:

[Open an issue](https://github.com/E5Anant/archeAI/issues)
[Submit a pull request](https://github.com/E5Anant/archeAI/pulls)

Check out our [Contribution Guidelines](https://github.com/E5Anant/archeAI/blob/main/CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the [MIT License](https://github.com/E5Anant/archeAI/blob/main/LICENSE).

## ⭐ Don't Forget to Star!

If you find ArcheAI helpful, please give us a star on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/E5Anant/archeAI.svg?style=social&label=Star)](https://github.com/E5Anant/archeAI)

</div>
