# Agent Class

The `Agent` class is designed to create an AI agent that can intelligently interact with various tools and an underlying LLM (Large Language Model). This class can dynamically manage tools, execute tasks, and generate responses based on provided inputs.

## Features

- **Dynamic Tool Management:** Add or remove tools as needed without restarting the agent.
- **Tool Selection:** Automatically selects the appropriate tool based on the task description.
- **JSON Response Handling:** Generates and processes JSON-based outputs for interaction with tools.
- **Async Tool Execution:** Executes tool functions asynchronously for efficient multitasking.
- **Verbose Mode:** Detailed logging for debugging and understanding the decision-making process.
- **LLM Integration:** Uses the GroqLLM for generating AI-driven responses.