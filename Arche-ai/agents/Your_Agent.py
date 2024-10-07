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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        task: str = "Hello, how are you?",
        skills: str = "Productive and helpful",
        verbose: bool = False,
        memory: bool = True, 
        memory_dir: str = "memories",
        max_memory_responses: int = 80,  
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

        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.memory = Memory(
            llm=self.llm,                # Pass the LLM object to Memory
            status=self.memory_enabled,   # Enable/Disable memory based on agent settings
            history_dir=self.memory_dir,    # Directory to store memory files 
            max_responses=max_memory_responses, # Max responses before summarization (adjust as needed) 
            update_file=self.memory_enabled,  # Update memory files if memory is enabled 
            memory_filepath=f"{self.name}_memory.txt",
            chat_filepath=f"{self.name}_chat.txt",
            system_prompt=f"You are {self.name}, {self.description}."  # Provide Memory with agent's persona
        )

        # Tool and Function Setup
        self.all_functions = [
            convert_function(tool.func.__name__, tool.description, **(tool.params or {})) for tool in self.tools
        ] + [
            convert_function(
                "llm_tool",
                "A default tool that returns AI-generated text responses using the context of previous tool calls. While it strives to be helpful and informative, please note it cannot access real-time information or execute actions in the real world.",
                query={'type': 'str', 'description': 'The query to be answered.'}
            )
        ]   

    def add_tool(self, tool: Tool):
        """Add a tool dynamically."""
        self.tools.append(tool)
        self.all_functions.append(convert_function(tool.func.__name__, tool.description, **(tool.params or {})))

    def remove_tool(self, tool_name: str):
        """Remove a tool dynamically."""
        self.tools = [tool for tool in self.tools if tool.func.__name__ != tool_name]
        self.all_functions = [func for func in self.all_functions if func['function']['name'] != tool_name]

    def _run_no_tool(self) -> str:
        """Handles user queries when no tools are available."""
        prompt = self.memory.gen_complete_prompt(self.task)
        self.llm.__init__(system_prompt=f"""
        **You are {self.name}, {self.description}.**

        ### OUTPUT STYLE:
        {self.expected_output}

        """, messages=[])
        result = self.llm.run(prompt)
        self.llm.reset()
        self.memory.update_chat_history("You", result, force=True)
        return result


    def _run_with_tools(self) -> str:
        # Initializing LLM with a universal prompt for tool chaining and handling errors
        self.llm.__init__(system_prompt=f"""
            **You are {self.name}, an advanced AI assistant capable of using multiple tools in sequence (tool chaining) to fulfill complex user requests. Your task is to manage the execution of tools, process their outputs, and pass them as inputs to subsequent tools when necessary.

**Outline for AI Tool Chaining**

#### I. Introduction
As an advanced AI assistant, I excel in executing complex user requests by seamlessly chaining multiple tools to achieve efficient and accurate results.

#### II. Tool Chaining Instructions
A. **Identifying Tool Sequences**
1. When a user request can't be fulfilled by a single tool, I scrutinize the request to identify the appropriate sequence of tools needed for its fulfillment.**
2. Upon recognition of the tool sequence, I initiate the chaining process.

B. **Executing the Sequence**
1. During tool chaining, I ensure that the output of each tool seamlessly transitions as the input for the subsequent tool.
2. If necessary, I adapt or reformat the output to align with the input requirements of the following tools.

#### III. Error Handling Instructions
I refrain from validating the logical output of one tool for its chaining to another, as this task is managed by the Compiler.

#### IV. Available Tools
The following tools are available for tool chaining: {{ self.all_functions }}.

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
A. **Example of Tool Chaining**
- **User Request:** "Get the current price of Bitcoin and convert it to USD."
- **Tool Chain Logic:**
   1. Utilize the web search tool to obtain the current price of Bitcoin.
   2. Employ the currency converter tool to convert the Bitcoin price to USD.
- **JSON Response Example:**

   {{
       "func_calling": [
           {{
               "tool_name": "web_search",
               "parameter": {{"query": "current price of Bitcoin"}},
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
- Only the `llm_tool` should be used for normal conversations.
- All responses should strictly adhere to the JSON format.
- Do not hallucinate tool names or capabilities. You can only use the tools provided. 
- If a tool requires specific information, make sure it is available in the context or from previous tool calls. 
- If no tool is suitable, use the 'llm_tool' for a conversational response.
- Avoid generating text and always guide the `llm_tool` responses from the user's perspective.
- The `llm_tool` should be used to generate text responses only.
- If a conversation query is doubtfull to you, give the query as is to the `llm_tool`.

#### Conclusion
Emphasize the significance of adhering to the outlined procedures to ensure the seamless execution of tool chaining, error management, and the successful handling of user requests.
            """,
            messages=[],
        )

        prompt = self.task
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

            # Check if any parameter value contains a placeholder
            parameters = call.get("parameter", {})
            for k, v in parameters.items():
                if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                    # Dynamic Tool Call Determination
                    next_tool_call = self._get_next_tool_call(results[tool_name], self.all_functions, self.task)
                    if next_tool_call:
                        # Update the current tool call with the intermediary LLM's suggestion
                        call["tool_name"] = next_tool_call["tool_name"]
                        call["parameter"] = next_tool_call["parameter"]

            try:
                tool_response = self._call_tool(call, results)  # Pass results to _call_tool
                results[tool_name] = tool_response  # Store output with tool_name instead of call_ID
                if self.verbose:
                    print(f"{Fore.GREEN}Tool {tool_name}:{Style.RESET_ALL} {tool_response}")
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}")
                results[tool_name] = f"Error: {e}"

        if self.memory_enabled:
            # Update memory directly with JSON tool results
            self.memory.update_chat_history("Tools", json.dumps(results, indent=2), force=True)

        return self._generate_summary(results)

    def _parse_and_fix_json(self, json_str: str) -> Dict | str:
        """Parses JSON string and attempts to fix common errors."""
        json_str = json_str.strip()
        if not json_str.startswith('{') or not json_str.endswith('}'):
            json_str = json_str[json_str.find('{'):json_str.rfind('}') + 1]

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
        context_str = "\n".join(
            [f"- Tool {call_id}: {output}" for call_id, output in tool_results.items()]
        ) if tool_results else "No previous tool calls."

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
            ### User Query: {query.get('query', '')} 
        """,
            messages=[],
        )

        prompt_context = (
            self.memory.gen_complete_prompt("") 
            if self.memory_enabled
            else ""
        )

        response = self.llm.run(prompt_context)
        self.llm.reset()
        return response

    def _generate_summary(self, results: Dict[str, str]) -> str:
        # Format tool results for the LLM
        results_formatted = "\n".join(
            [f"Tool {call_id}: {output}" for call_id, output in results.items()]
        )
        prompt = f"User Query:\n{self.task}\n\nTool Results:\n{results_formatted}\n\n"

        if self.memory_enabled:
            prompt = self.memory.gen_complete_prompt(prompt)

        self.llm.__init__(
            system_prompt=f"""
            You are {self.name}, an AI agent. {self.description}.
            You have been provided with the results of various tools used to process a user's query.
            Your task is to use these tool results to provide the best possible response to the user. 

            ### OUTPUT STYLE:
            {self.expected_output}

            ## Instructions:
            - **Do not create information that is not supported by tool outputs.**
            - Always base your response only on the outputs from tools, not on assumptions.
            - If the tool results are insufficient to fully answer the user's query, explain the limitations.
        """,
            messages=[],
        )

        summary = self.llm.run(prompt)

        if self.verbose:
            print("")
            print("Final Response:")
            print(summary)
        if self.memory_enabled:
            self.memory.update_chat_history("You", summary, force=True)

        return summary

    def _get_next_tool_call(self, previous_tool_output: str, all_tools: list, user_task: str) -> dict:
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

        ### Rules:
        - **Do not invent tools or steps.** If no valid next tool exists, return an empty JSON object.
        - Your decision must be based on the previous tool's output and the user’s task.
        - Provide your response in the JSON format:
        {{
            "tool_name": "<tool_name>",
            "parameter": {{"<param_name>": "<param_value>"}}
        }}
        If no tool is appropriate, return:
        {{
            "tool_name": "",
            "parameter": {{}}
        }}
        """

        self.llm.__init__(system_prompt="You are an assistant helping to choose the best tool...")  # Initialize your LLM
        response = self.llm.run(prompt)  # Assuming your LLM has a .run(prompt) method
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