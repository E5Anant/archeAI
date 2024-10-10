```python

import json
import re
import os
import logging
from archeai.llms import GroqLLM  # Your LLM classes
from archeai.tools import Tool
from typing import Type, List, Dict, Any
from colorama import Fore, Style
from archeai.memory import Memory  # Import the Memory class

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
                            "enum": "string",  # Enum should be a string with specific values
                            "array": "array",
                            "list": "array",
                            "dict": "object",  # 'dict' maps to 'object' in JSON schema
                            "dictionary": "object",  # Similarly for 'dictionary'
                            "object": "object",
                            "obj": "object",
                        }.get(
                            param_info.get("type", "string").lower(), "string"
                        ),
                        "description": param_info.get(
                            "description",
                            f"Description for {param_name} is missing.",
                        ),
                        **(
                            {"enum": param_info["options"]}
                            if param_info.get("type", "").lower() == "enum"
                            else {}
                        ),
                        **(
                            {"default": param_info["default"]}
                            if "default" in param_info
                            else {}
                        ),
                    }
                    for param_name, param_info in params.items()
                },
                "required": [
                    param_name
                    for param_name, param_info in params.items()
                    if param_info.get("required", False)
                ],
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
        task: str = "Hello, how are you?",
        skills: str = "Productive and helpful",
        verbose: bool = False,
        memory: bool = True,
        memory_dir: str = "memories",
        max_memory_responses: int = 12,
    ) -> None:
        """
        Args:
        llm (Type[GroqLLM]): The LLM class.
        tools (List[Tool], optional): A list of tools. Defaults to [].
        name (str, optional): The name of the agent. Defaults to "Agent".
        description (str, optional): The description of the agent. Defaults to "A helpful AI agent.".
        expected_output (str, optional): The expected output format. Defaults to "Concise and informative text.".
        task (str, optional): The task of the agent. Defaults to "Hello, how are you?".
        skills (str, optional): The skills of the agent. Defaults to "Productive and helpful".
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        memory (bool, optional): Whether to enable memory for the agent. Defaults to True.
        memory_dir (str, optional): The directory to store memory files. Defaults to "memories".
        max_memory_responses (int, optional): The maximum number of responses to store in memory. Defaults to 12.
        """

        self.llm = llm
        self.tools = tools
        self.name = name
        self.description = description
        self.expected_output = expected_output
        self.task = task
        self.skills = skills
        self.verbose = verbose
        self.memory_enabled = memory

        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.memory = Memory(
            llm=self.llm,  # Pass the LLM object to Memory
            status=self.memory_enabled,  # Enable/Disable memory based on agent settings
            assistant_name=self.name,  # Directory to store memory files
            history_dir=self.memory_dir,
            max_responses=max_memory_responses,  # Max responses before summarization (adjust as needed)
            update_file=self.memory_enabled,  # Update memory files if memory is enabled
            memory_filepath=f"{self.name}_memory.txt",
            chat_filepath=f"{self.name}_chat.txt",
            system_prompt=f"You are {self.name}, {self.description}.",  # Provide Memory with agent's persona
        )

        # Tool and Function Setup
        self.all_functions = [
            convert_function(
                tool.func.__name__, tool.description, **(tool.params if tool.params else {})
            )
            for tool in self.tools
        ] + [
            # llm and direct answer initialization
        ]

    def _run_no_tool(self) -> str:
        """Handles user queries when no tools are available."""
        prompt = self.memory.gen_complete_prompt(self.task)
        self.llm.__init__(
            system_prompt=f"""
        **You are {self.name}, {self.description}.**

        ### OUTPUT STYLE:
        {self.expected_output}

        """,
            messages=[],
        )
        result = self.llm.run(prompt)
        self.llm.reset()
        self.memory.update_chat_history(self.name, result, force=True)
        return result

    def _run_with_tools(self) -> str:
        # Initializing LLM with a universal prompt for tool chaining and handling errors
        self.llm.__init__(
            system_prompt=f"""
            **You are {self.name}, an advanced AI assistant capable of using multiple tools in sequence (tool chaining) to fulfill complex user requests. Your task ....

....

#### IV. Available Tools
The following tools are available for tool chaining:
{ self.all_functions }.

#### V. JSON Format for Tool Calls
   {{
       "func_calling": [
           {{
               "tool_name": "<tool_name>",
               "parameter": {{"<param_name>": "<param_value>"}},
               "call_ID": ""
           }}
       ]
   }}


#### VI. User Request Examples
......
- **JSON Response Example:**

   {{
       "func_calling": [
           {{
               ...
               ....
               "call_ID": "1"
           }},
           {{
               "tool_name": "currency_converter",
               "parameter": {{"amount": "{{1.output}}", "from": "BTC", "to": "USD"}},
               "call_ID": "2"
           }}
       ]
   }}


B. **Example of Normal Conversation**
- **User Request:** "Hello! I am John."
- **JSON Response Example:**

   {{
       "func_calling": [
           {{
               "tool_name": "llm_tool",
               "parameter": {{"query": "Hello! I am John."}},
               "call_ID": "1"
           }}
       ]
   }}


#### VII. Important Notes
....Instruction 

#### Conclusion
....
            """,
            messages=[],
        )

        if self.memory_enabled:
            prompt = self.memory.gen_complete_prompt(self.task)
            response = self.llm.run(prompt).strip()
        else:
            response = self.llm.run(self.task).strip()
        self.llm.reset()

        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        action = self._parse_and_fix_json(response)
        if isinstance(action, str):
            return action

        results = {}

        # Handle llm_tool separately to ensure its output is available for the next tool call
        if "llm_tool" in action.get("func_calling", []):
            for call in action.get("func_calling", []):
                if call["tool_name"] == "llm_tool":
                    try:
                        tool_response = self._call_tool(call, results)
                        results = {"llm_tool": tool_response}
                    except Exception as e:
                        if self.verbose:
                            print(
                                f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}"
                            )
                        results = {"llm_tool": f"Error: {e}"}
                    break  # Stop after handling llm_tool

        for i, call in enumerate(action.get("func_calling", [])):
            tool_name = call["tool_name"]

            # Check if any parameter value contains a placeholder
            parameters = call.get("parameter", {})
            for k, v in parameters.items():
                if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                    for function in self.all_functions:
                        if function["function"]["name"] == tool_name:
                            func = function
                            break  # Stop after finding the desired tool
                                        # Dynamic Tool Call Determination
                    print(tool_name)
                    next_tool_call = self._get_next_tool_call(
                        results[tool_name], func, self.task
                    )
                    if next_tool_call:
                        # Update the current tool call with the intermediary LLM's suggestion
                        call["tool_name"] = next_tool_call["tool_name"]
                        call["parameter"] = next_tool_call["parameter"]

            try:
                tool_response = self._call_tool(
                    call, results
                )  # Pass results to _call_tool
                results[tool_name] = (
                    tool_response  # Store output with tool_name instead of call_ID
                )
                if self.verbose:
                    print(
                        f"{Fore.GREEN}Tool {tool_name}:{Style.RESET_ALL} {tool_response}"
                    )
            except Exception as e:
                if self.verbose:
                    print(
                        f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}"
                    )
                results[tool_name] = f"Error: {e}"

        if self.memory_enabled:
            # Update memory directly with JSON tool results
            self.memory.update_chat_history(
                "Tools", json.dumps(results, indent=2), force=True
            )

        return self._generate_summary(results)

    def _parse_and_fix_json(self, json_str: str) -> Dict | str:
        """Parses JSON string and attempts to fix common errors."""
        pass

    def _direct_answer(self, query, tool_results):
        # Use this function to return answers directly when applicable without external LLM
        # It checks if a direct answer can be provided from the query or past results
        answer = query.get("query", "").lower()

        if answer:
            return f"Direct Answer: {answer}"

        # In case no direct answer is feasible, fallback to llm_tool or other tools
        return self._process_llm_tool(query, tool_results)

    def _call_tool(self, call, results):
        """Handles tool invocation based on the call object."""
        tool_name = call.get(
            "tool_name", ""
        ).lower()  # Normalize tool name to lowercase
        query = call.get("parameter", {})

        if not tool_name:
            raise ValueError("No tool name provided in the call.")

        # Handle special tools with custom logic
        if tool_name == "direct_answer":
            return self._direct_answer(query, results)

        if tool_name == "llm_tool":
            return self._process_llm_tool(query, results)

        # Find the tool by matching function name
        tool = next(
            (t for t in self.tools if t.func.__name__.lower() == tool_name),
            None,
        )
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        # Handle tool execution with or without parameters
        try:
            if tool.params:  # If tool expects parameters
                tool_response = tool.func(**query)
            else:
                tool_response = tool.func()
        except Exception as e:
            raise ValueError(f"Error executing tool '{tool_name}': {str(e)}")

        return tool_response if tool.returns_value else "Action completed."

    def _process_llm_tool(self, query, tool_results):
        """Handles LLM tool invocation based on the query and tool_results."""
        # Check if query is a dictionary and extract the 'query'
        if isinstance(query, dict):
            query = query.get("query", "")

        # Ensure query is a string and process it
        if isinstance(query, str):
            query = query.lower()  # Example of processing
        else:
            raise ValueError(
                "The query must be a string or a dictionary containing a 'query'."
            )

        context_str = (
            "\n".join(
                [
                    f"- Tool {call_id}: {output}"
                    for call_id, output in tool_results.items()
                ]
            )
            if tool_results
            else "No previous tool calls."
        )

        self.llm.__init__(
            system_prompt=f"""
            **You are {self.name}, {self.description}.**
            You are provided with the context of the conversation and, if applicable, the results of previous tool calls. 
            Your task is to craft a helpful and informative response to the user's query.

            ### Instructions:
            - **Do not fabricate information or claim to have access to data that is not provided.**
            - **Do not refer to specific previous turns or tool calls.** The user only sees your current response.

            ### Context: 
            {context_str}
            ### User Query: {query} 
            """,
            messages=[],
        )

        prompt_context = (
            self.memory.gen_complete_prompt(query)
            if self.memory_enabled
            else ""
        )

        response = self.llm.run(prompt_context)
        self.llm.reset()
        return response

    def _generate_summary(self, results: Dict[str, str]) -> str:
        # generates summary using llm logic removed to make the code appear shorter to you
        pass

    def _get_next_tool_call(
        self, previous_tool_output: str, all_tools, user_task: str
    ) -> dict:
        """
        Determines the next tool to call based on previous tool output and user task.

        Args:
            previous_tool_output (str): The output from the previous tool call.
            all_tools (list): A list of available tools.
            user_task (str): The user's original task.

        Returns:
            dict: A dictionary containing the next tool name and parameters.
        """

        # Construct the prompt for the intermediary LLM
        prompt = f"""
        You are an AI assistant. Your task is to determine the next tool to call based on the previous tool's output and the user's task. 

        ## Previous_tool_ouput:
        {previous_tool_output}

        ## User Task:
        {user_task}

        ## Tool to call:
        {all_tools}

        ### Rules:
        - **Do not invent tools or steps.** If no valid next tool exists, return an empty JSON object.
        - Your decision must be based on the previous tool's output and the user’s task.
        - Provide your response in the JSON format:
        {{
            "tool_name": "<tool_name>",
            "parameter": {{"<param_name>": "<param_value>"}}
        }}
        """

        self.llm.__init__(
            system_prompt="You are an assistant helping to choose the best tool..."
        )  # Initialize your LLM
        response = self.llm.run(
            prompt
        )  # Assuming your LLM has a .run(prompt) method
        self.llm.reset()  # Reset your LLM

        if self.verbose:
            print("Intermediary LLM response:")
            print(response)

        # Parse the JSON response
        try:
            response = self._parse_and_fix_json(response)
            return response
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from intermediary LLM: {e}")
            return {}  # Return an empty dictionary on error

    def rollout(self) -> str:
        """
        Executes the agent's workflow to process the task and return a response.
        """
        self.memory.update_chat_history("User", self.task, force=True)
        self.llm.reset()
        if not self.tools:
            return self._run_no_tool() if self.task else "No task provided."
        return self._run_with_tools()

```

### Output:

Raw LLM Response: ```json
{
 "func_calling": [
  {
   "tool_name": "get_current_time",
   "parameter": {},
   "call_ID": "1"
  },
  {
   "tool_name": "llm_tool",
   "parameter": {
    "query": "What should we do at {1.output}?"
   },
   "call_ID": "2"
  },
  {
   "tool_name": "write_to_file",
   "parameter": {
    "filename": "time.txt",
    "content": "The current time is {1.output}.\n\n{2.output}"
   },
   "call_ID": "3"
  }
 ]
}
```
Tool get_current_time: 12:43:10
llm_tool
Traceback (most recent call last):
  File "d:\Arche-ai\app.py", line 99, in <module>
    Chatbot.rollout()
  File "d:\Arche-ai\archeai\agent.py", line 596, in rollout
    return self._run_with_tools()
  File "d:\Arche-ai\archeai\agent.py", line 334, in _run_with_tools
    results[tool_name], func, self.task
KeyError: 'llm_tool