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
            convert_function(
                "llm_tool",
                "A default tool that returns AI-generated text responses using the context of previous tool calls. While it strives to be helpful and informative, please note it cannot access real-time information or execute actions in the real world.",
                query={
                    "type": "str",
                    "description": "The query to be answered.",
                },
            ),
            convert_function(
                "direct_answer",  # Tool for direct answers
                "Use this tool when you can directly answer the user's query from the context or your knowledge. Avoid using this tool if the user's query requires specific actions or if you need to gather external information.",
                answer={
                    "type": "str",
                    "description": "The direct answer to the user's query.",
                },
            ),
        ]

    def add_tool(self, tool: Tool):
        """Add a tool dynamically."""
        self.tools.append(tool)
        self.all_functions.append(
            convert_function(
                tool.func.__name__, tool.description, **(tool.params or {})
            )
        )

    def remove_tool(self, tool_name: str):
        """Remove a tool dynamically."""
        self.tools = [
            tool for tool in self.tools if tool.func.__name__ != tool_name
        ]
        self.all_functions = [
            func
            for func in self.all_functions
            if func["function"]["name"] != tool_name
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
- **Use the 'direct_answer' tool** when you can directly answer the user's question without using any other tools.
- **Do not refer to specific previous turns or tool calls.** The user only sees your current response.
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
        for i, call in enumerate(action.get("func_calling", [])):
            tool_name = call["tool_name"]

            # Check if any parameter value contains a placeholder
            parameters = call.get("parameter", {})
            for k, v in parameters.items():
                if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                    # Dynamic Tool Call Determination
                    next_tool_call = self._get_next_tool_call(
                        results[tool_name], self.all_functions, self.task
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
        json_str = json_str.strip()
        if not json_str.startswith("{") or not json_str.endswith("}"):
            json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"{Fore.RED}JSON Error:{Style.RESET_ALL} {e}")

            # Common JSON fixes
            json_str = json_str.replace("'", '"')  # Replace single quotes
            json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas
            json_str = re.sub(r"{\s*,", "{", json_str)  # Remove leading commas
            json_str = re.sub(
                r"\s*,\s*", ",", json_str
            )  # Remove whitespaces around commas

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return f"Error: Could not parse JSON - {e}"

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
            - **Do not refer to specific previous turns or tool calls.** The user only sees your current response.
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
            self.memory.update_chat_history(self.name, summary, force=True)

        return summary

    def _get_next_tool_call(
        self, previous_tool_output: str, all_tools: list, user_task: str
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
