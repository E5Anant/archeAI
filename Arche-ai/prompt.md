You are an expert Python developer familiar with large language models (LLMs) and agent-based systems. You are tasked with improving the code of an `Agent` class that uses tools and an LLM to complete tasks. The current code is provided below:

```python
import json
import re
import os
import time
import threading
import logging
from llms import GroqLLM, Gemini, Cohere  # Your LLM classes
from tools import Tool
from typing import Type, List, Optional, Dict, Any
from colorama import Fore, Style
from memory import Memory  # Import the Memory class
import concurrent.futures

def convert_function(func_name, description, **params):
    """Converts function info to JSON schema, handling missing params."""
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": {
                            "string": "string",
                            "str": "string",
                            "number": "number",
                            "num": "number",
                            "int": "integer",
                            "integer": "integer",
                            "boolean": "boolean",
                            "bool": "boolean",
                            "enum": "enum",
                            "array": "array",
                            "list": "array",
                            "dict": "dictionary",
                            "dictionary": "dictionary",
                            "object": "object",
                            "obj": "object",
                        }.get(param_info.get("type", "string").lower(), "string"),
                        "description": param_info.get("description", f"Description for {param_name} is missing."),
                        **({"enum": param_info["options"]} if param_info.get("type", "").lower() == "enum" else {}),
                        **({"default": param_info["default"]} if "default" in param_info else {}),
                    }
                    for param_name, param_info in params.items()
                },
                "required": [param_name for param_name, param_info in params.items() if param_info.get("required", False)],
            },
        },
    }


class Agent:
    def __init__(
        self,
        llm: Type[GroqLLM],
        tools: List[Tool] = [],
        name: str = "Agent",
        description: str = "A helpful AI agent.",
        expected_output: str = "Concise and informative text.",
        task: str = "Ask me a question or give me a task.",
        skills: str = "Productive and helpful",
        verbose: bool = False,
        memory: bool = True,  # Enable/Disable memory
        memory_dir: str = "memories",  # Directory to store memories
        max_memory_tokens: int = 800,  # Max tokens for memory
        memory_history_offset:int = 10250,
        update_memory_files:bool = True
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.name = name
        self.description = description
        self.expected_output = expected_output
        self.task = task
        self.skills = skills
        self.verbose = verbose
        self.memory_enabled = memory
        self.offset = memory_history_offset
        self.update_file = update_memory_files
        # Memory Setup
        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.memory = Memory(
            llm=self.llm,  # Pass the LLM instance for memory summarization
            status=self.memory_enabled,
            max_tokens=max_memory_tokens,
            memory_filepath=os.path.join(self.memory_dir, f"{self.name}_memory.txt"),
            chat_filepath=os.path.join(self.memory_dir, f"{self.name}_chat.txt"),
            system_prompt=f"You are {self.name}, {description}.",
            history_offset=self.offset,
            update_file=self.update_file,
        )

        self.all_functions = [
            convert_function(tool.func.__name__, tool.description, **(tool.params or {})) for tool in self.tools
        ] + [convert_function("llm_tool", "A default tool that returns AI-generated text responses using the context of previous tool calls for the given query. It cannot answer real-time queries due to a knowledge cutoff of October 2019.", **{})]


    def add_tool(self, tool: Tool):
        """Add a tool dynamically."""
        self.tools.append(tool)
        self.all_functions.append(convert_function(tool.func.__name__, tool.description, **(tool.params or {})))

    def remove_tool(self, tool_name: str):
        """Remove a tool dynamically."""
        self.tools = [tool for tool in self.tools if tool.func.__name__ != tool_name]
        self.all_functions = [func for func in self.all_functions if func['function']['name'] != tool_name]

    def _run_no_tool(self) -> str:
        prompt = self.memory.gen_complete_prompt(self.task) if self.memory_enabled else self.task
        self.llm.__init__(system_prompt=f"""
        You are {self.name}, {self.description}.
        ### OUTPUT STYLE:
        {self.expected_output}
        ***If output style not mentioned, generate in markdown format.***
        """, messages=[])
        result = self.llm.run(prompt)
        self.llm.reset()
        
        if self.memory_enabled:
            # Automatically update memory using the memory class
            self.memory.update_chat_history(self.name, result, force=True)  # Update memory
        
        return result   

    def _run_with_tools(self) -> str:
        self.llm.__init__(system_prompt=f"""
        You are an AI assistant that generates JSON responses based on provided tools.

        **Reasoning:**
        1. Determine if the task needs real-time data (e.g., weather, calculations) or if it's a simple question.
        2. If it needs data or calculations, choose the right tool and its parameters.
        3. If it's a simple question, answer directly using `llm_tool`.

        **Tools:**
        {self.all_functions}

        **Instructions:**
        1. Understand the task.
        2. Identify tool parameters.
        3. Respond ONLY with a JSON object containing "tool_name", "parameter" and "call_ID".
        4. No text outside the JSON.
        5. To use the output of one tool as input to another tool you can use the following format: {{<call_ID>.output}}
            For example, if the output of the tool with call_ID 1 is "Hello" and you want to pass it to another tool you can use {{1.output}} which will be replaced with "Hello"

        **JSON Format:**
        {{
            "func_calling": [
                {{
                    "tool_name": "<tool_name>",
                    "parameter": {{<param_name>: "<param_value>"}},
                    "call_ID": "<call_ID>"
                }}
            ]
        }}

        **For tools without parameters:**
        {{
            "func_calling": [
                {{
                    "tool_name": "<tool_name>",
                    "parameter": {{}}
                    "call_ID": "<call_ID>"
                }}
            ]
        }}

        **Example (with parameters):**
        Task: Get New York weather.
        Response:
        {{
            "func_calling": [
                {{
                    "tool_name": "weather_tool",
                    "parameter": {{"query": "New York"}},
                    "call_ID": "1"
                }}
            ]
        }}

        **Example (without parameters):**
        Task: What time is it?
        Response:
        {{
            "func_calling": [
                {{
                    "tool_name": "time_tool",
                    "parameter": {{}}
                    "call_ID": "1"
                }}
            ]
        }}
        **Example (Use output of one tool to another tool):**
        Task: Get the current time and tell me the hour.
        Response:
        {{
            "func_calling": [
                {{
                    "tool_name": "get_time",
                    "parameter": {{}},
                    "call_ID": "1"
                }},
                {{
                    "tool_name": "llm_tool",
                    "parameter": {{'query': "Tell me the hour from this time: {{1.output}}"}}
                    "call_ID": "2"
                }}
            ]
        }}
        **Example (with llm_tool):**
        Task: Who are you?
        Response:
        {{
            "func_calling": [
                {{
                    "tool_name": "llm_tool",
                    "parameter": {{'query': "Who are you?"}}
                    "call_ID": "1"
                }}
            ]
        }}

        ***How To distribute task into sub-tasks:-***
        Task: What is today's date and time.
        Response:
        {{
            "func_calling": [
                {{
                    "tool_name": "get_date",
                    "parameter": {{}}
                    "call_ID": "1"
                }},
                {{
                    "tool_name": "get_time",
                    "parameter": {{}}
                    "call_ID": "2"
                }}
            ]
        }}

        **REMEMBER, Important:**
        - Tool and parameter names vary based on the provided tools.
        - Always use tools; `llm_tool` is for basic communication only.
        - Break a single task into sub-tasks and can call a single tool two times to accomplish the task with accuracy. 
        """, messages=[])
        
        prompt = self.memory.gen_complete_prompt(self.task) if self.memory_enabled else self.task
        response = self.llm.run(prompt).strip()
        self.llm.reset()

        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        action = self._parse_and_fix_json(response)
        if isinstance(action, str):
            return action
        
        results = {}
        for i, call in enumerate(action.get("func_calling", [])):
            tool_name = call["tool_name"]
            
            #Substitute the parameters
            parameters = call.get("parameter", {})
            for k, v in parameters.items():
                if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                    for prev_call in action.get("func_calling", [])[:i]:
                        if f"{{{prev_call['call_ID']}.output}}" in v:
                            parameters[k] = v.replace(f"{{{prev_call['call_ID']}.output}}", str(results[prev_call['call_ID']])) # Convert to string 
            try:
                tool_response = self._call_tool(call, results) # Pass results to _call_tool
                results[call['call_ID']] = tool_response  # Store output with call_ID
                if self.verbose:
                    print(f"{Fore.GREEN}Tool {tool_name}:{Style.RESET_ALL} {tool_response}")
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}")
                results[call['call_ID']] = f"Error: {e}"

        if self.memory_enabled:
            # Update memory directly with JSON tool results
            self.memory.update_chat_history("Tools", json.dumps(results, indent=2), force=True)

        return self._generate_summary(results)

  
    def _parse_and_fix_json(self, json_str: str) -> Dict | str:
        """Parses JSON string and attempts to fix common errors."""

        # Remove any text before the starting '{' and after the ending '}'
        json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"{Fore.RED}JSON Error:{Style.RESET_ALL} {e}")

            # Common JSON fixes
            json_str = json_str.replace("'", "\"")  # Replace single quotes
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r'{\s*,', '{', json_str)  # Remove leading commas
            json_str = re.sub(r'\s*,\s*', ',', json_str)  # Remove whitespaces around commas

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return f"Error: Could not parse JSON - {e}" 

    def _call_tool(self, call, results):
        tool_name = call["tool_name"]
        query = call.get("parameter", {})  

        # Find the tool
        tool = next((t for t in self.tools if t.func.__name__ == tool_name), None)

        if tool_name.lower() == "llm_tool":
            return self._process_llm_tool(query, results)

        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        # Handle tool parameters
        if tool.params and isinstance(query, dict):
            try:
                tool_response = tool.func(**query)
            except TypeError as e:
                if "unexpected keyword argument" in str(e) or "missing 1 required positional argument" in str(e):
                    tool_response = tool.func(*query.values())
                else:
                    raise e
        else:
            tool_response = tool.func()

        return tool_response if tool.returns_value else "Action completed."

    def _process_llm_tool(self, query, tool_results):
        prompt_context = f"Previous tool results:\n{tool_results}\n\nQuery: {query}"

        self.llm.__init__(system_prompt=f"""
            You are {self.name}, an AI agent. {self.description}.
            You will receive information from previous tool calls. Use this information to answer the query.

            **Important:**
            - You CANNOT directly interact with tools or files.
            - Your role is ONLY to generate text responses based on the given information. 
            - DO NOT generate content that should be written to files. 
            - DO NOT assume that a file has been written to, even if the query asks for it.
            - DO NOT generate code or try to call any code interpreter.

            ### OUTPUT STYLE:
            {self.expected_output}

            ## Instructions:
            - Respond clearly and concisely, using the provided tool results.
            - If no output style is specified, respond in the most appropriate way.
        """, messages=[])

        response = self.llm.run(prompt_context)
        self.llm.reset()
        return response

    def _generate_summary(self, results: Dict[str, str]) -> str:
        prompt = f"[QUERY]\n{self.task}\n\n[TOOLS]\n{results}"
        if self.memory_enabled:
            prompt = self.memory.gen_complete_prompt(prompt)
        
        self.llm.__init__(system_prompt=f"""
            You are {self.name}, an AI agent. {self.description}.
            You will receive information from previous tool calls. Use this information to answer the query.

            **Important:**
            - You CANNOT directly interact with tools or files.
            - Your role is ONLY to generate text responses based on the given information. 
            - DO NOT generate content that should be written to files. 
            - DO NOT assume that a file has been written to, even if the query asks for it.
            - DO NOT generate code or try to call any code interpreter.

            ### OUTPUT STYLE:
            {self.expected_output}

            ## Instructions:
            - Respond clearly and concisely, using the provided tool results.
            - If no output style is specified, respond in the most appropriate way.
        """, messages=[])
        
        summary = self.llm.run(prompt)
        if self.verbose:
            print("Final Response:")
            print(summary)
        
        if self.memory_enabled:
            # Update memory with final summary output
            self.memory.update_chat_history(self.name, summary, force=True)
        
        return summary

    def rollout(self) -> str:
        self.memory.add_message("User", self.task)
        self.llm.reset()
        if not self.tools:
            return self._run_no_tool() if self.task else "No task provided."
        return self._run_with_tools()
```

**Problem**: The current implementation requires the user to manually chain tool outputs using placeholders like {{1.output}}. This is not user-friendly and requires knowledge of internal call IDs.

**Solution**: We can make the agent smarter by introducing an intermediary LLM. This LLM will analyze the output of each tool call and determine the next best tool to use, along with its parameters, based on the user's original task and the available tools.

Your task is to modify the provided code according to these detailed instructions:

1. Create a new function called _get_next_tool_call:

    **Inputs:**

        - ***previous_tool_output***: The output (string) from the previous tool call.

        - ***all_tools***: The list of all available tools, formatted as in the original all_functions variable.

        - ***user_task***: The user's original task (string).`

    **Functionality:**

    - This function will interact with an intermediary LLM (you don't need to implement the LLM interaction itself, assume a function `call_intermediary_llm(prompt)` is available).

    - Construct a clear and concise prompt for the intermediary LLM. This prompt should include:

        - The previous tool's output.

        - The user's original task.

        - The descriptions of all available tools (name, description, parameters).

        - Instructions to analyze the provided information and determine the most logical next tool to call.

        - Instructions to extract relevant information from the previous tool's output to be used as parameters for the next tool call.

        - Instructions to output its suggestion in the following JSON format:

            ```json
            {
                [
                    {
                        "parameter": {'<param_name>': "<param_value>"},
                    }
                ]
            }

            ```
        - Call `call_intermediary_llm(prompt)` to get the JSON response.

        - Parse the JSON response and return a dictionary with the tool_name and parameter for the next tool call.

2. Modify the `_run_with_tools` method:

    - Inside the loop that iterates through the func_calling list, instead of directly substituting placeholders like {{1.output}}:

        - Detect if a parameter value contains a placeholder (e.g., using regular expressions).

        - If a placeholder is found:

            - Call the `_get_next_tool_call` function, passing the previous tool's output, the all_functions list, and the user's task (self.task).

            - Use the returned dictionary (containing tool_name and parameter) to overwrite the current tool call in the func_calling list. This ensures the agent uses the intermediary LLM's suggested tool and parameters.

    - Ensure that the rest of the _run_with_tools logic, including tool execution and result handling, remains compatible with these changes.

### Additional Considerations:

- Maintain the existing code structure and JSON formats as much as possible.

- Use clear variable names and add comments to explain the logic behind your changes.

- Prioritize code readability and maintainability.