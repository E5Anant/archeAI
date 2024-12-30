import json
import re
import os
from archeai.llms import GroqLLM  # Your LLM classes
from archeai.tools import Tool
from typing import Type, List, Dict, Any
from colorama import Fore, Style
import colorama
from rich.console import Console
from rich.panel import Panel
from rich import box

# Configure logging

colorama.init(autoreset=True)

console = Console()

def box_print(title: str, content: str, color: str):
    content = "This is the content inside the box."
    panel = Panel(content, title=title, box=box.SIMPLE, border_style=color,)
    console.print(panel)

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
        identity: str = "Agent",
        description: str = "A helpful AI agent.",
        expected_output: str = "Concise and informative text.",
        verbose: bool = False,
        ask_user: bool = True,
        max_iterations: int = 3,  # Add max_iterations parameter
        check_response_validity: bool = False,
        output_file: str = None,
    ) -> None:
        """
        Initialize the Agent class.

        Parameters
        ----------
        llm : Type[GroqLLM]
            The language model used by the agent.
        tools : List[Tool], optional
            A list of tools that the agent can use, by default []
        identity : str, optional
            The name of the agent, by default "Agent"
        description : str, optional
            A description of the agent, by default "A helpful AI agent."
        expected_output : str, optional
            The expected output of the agent, by default "Concise and informative text."
        verbose : bool, optional
            Whether the agent should be verbose, by default False
        memory : bool, optional
            Whether the agent should use memory, by default True
        memory_dir : str, optional
            The directory where the agent should store memory, by default "memories"
        max_chat_responses : int, optional
            The maximum number of chat responses before summarizing in memory, by default 12
        ask_user : bool, optional
            Whether the agent should ask the user for input, by default True
        max_summary_entries : int, optional
            The maximum number of summary entries to store in memory before summarizing, by default 5
        max_iterations : int, optional
            The maximum number of iterations, by default 3
        allow_full_delegation : bool, optional
            Whether the agent should allow full delegation switching to True would give all the previous responses from different agents. Switching to False would only allow the last response, by default False
        output_file : str, optional
            The name of the output file to store agent's response, by default None
        check_response_validity : bool, optional
            Whether the agent should check response validity, by default True
        """

        self.llm = llm
        self.tools = tools
        self.identity = identity
        self.description = description
        self.expected_output = expected_output
        self.ask_user = ask_user
        self.file = output_file
        self.objective = None
        self.cache_dir = None
        self.verbose = verbose
        self.max_iterations = max_iterations  # Assign to the instance variable
        self.check_validity = check_response_validity

        if self.ask_user:
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
                    "direct_answer",
                    "Use this tool when you can directly answer the user's query from the context or your knowledge. Avoid using this tool if the user's query requires specific actions or if you need to gather external information.",
                    query={
                        "type": "str",
                        "description": "The direct answer to the user's query.",
                    },
                ),
                convert_function(
                    "ask_user",
                    "Use this tool when you need to ask the user for input. Avoid using this tool if the user's query requires specific actions or if you need to gather external information.",
                    query={
                        "type": "str",
                        "description": "The query to be asked.",
                    },
                ),
            ]
        else:
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
                    "direct_answer",
                    "Use this tool when you can directly answer the user's query from the context or your knowledge. Avoid using this tool if the user's query requires specific actions or if you need to gather external information.",
                    query={
                        "type": "str",
                        "description": "The direct answer to the user's query.",
                    },
                )
            ]
        self.tool_info_for_validation = [
            {"name": tool.func.__name__, "description": tool.description} for tool in self.tools
        ]

    def _run_no_tool(self) -> str:
        for current_iteration in range(self.max_iterations):  # Iteration loop

            print(f"{Fore.CYAN}Iteration: {current_iteration + 1}{Style.RESET_ALL}")
            """Handles user queries when no tools are available."""
            prompt = self.objective
            self.llm.__init__(
                system_prompt=f"""
            **You are {self.identity}, {self.description}.**

            #### EXPECTED OUTPUT:
            {self.expected_output}

            #### QUERY:
            {self.objective}
            """,
                messages=[],
            )
            result = self.llm.run(prompt)
            self.llm.reset()
            if self._is_response_valid(result, {"response": result}):  # Implement your validation logic
                print(f"{Fore.GREEN}Response is valid, stopping iterations.{Style.RESET_ALL}")
                return result
            else:
                print(f"{Fore.YELLOW}Response is not valid, retrying...{Style.RESET_ALL}")
        return result
    
    def _ask_user(self, query: str) -> str:
        user_input = input(f"{query}: ")
        return user_input

    def _run_with_tools(self) -> str:
        for current_iteration in range(self.max_iterations):  # Iteration loop
            print(f"{Fore.CYAN}Iteration: {current_iteration + 1}{Style.RESET_ALL}")

            # Initializing LLM with an improved system prompt
            self.llm.__init__(system_prompt=f'''
### Role Description:
You are the JSON instructor for the AI agent '{self.identity}', an advanced AI model capable of using multiple tools in sequence to fulfill complex user requests.

### Primary Responsibilities:
1. Manage the execution of tools.
2. Process tool outputs.
3. Pass outputs as inputs to subsequent tools when necessary.

### Key Capabilities:
- Execute complex user requests by chaining multiple tools.
- Identify and implement appropriate tool sequences.
- Adapt tool outputs for subsequent tool inputs.
- Handle errors and unexpected situations gracefully.

### Available Tools:
The following tools are available for tool chaining:
{self.all_functions}.

### JSON Format for Tool Calls:
```json
{{
    "func_calling": [
        {{
            "tool_name": "<tool_name>",
            "parameter": {{"<param_name>": "<param_value>"}},
            "call_ID": "",
            "thought": "A brief description of the tool call or the performed action."
        }}
    ]
}}
```

### User Request Examples &amp; Tool Chain Logic:

#### A. Example of Tool Chaining:
**User Request:** "Get the current price of Bitcoin and convert it to USD."

#### Tool Chain Logic:
1. Utilize the web search tool to obtain the current price of Bitcoin.
2. Employ the currency converter tool to convert the Bitcoin price to USD.

**JSON Response Example:**
```json
{{
    "func_calling": [
        {{
            "tool_name": "web_search",
            "parameter": {{"query": "current price of Bitcoin"}},
            "call_ID": "1",
            "thought": "Searching the web for the current price of Bitcoin using the web_search tool."
        }},
        {{
            "tool_name": "currency_converter",
            "parameter": {{"amount": "{{1.output}}", "from": "BTC", "to": "USD"}},
            "call_ID": "2",
            "thought": "Converting the Bitcoin price to USD using the currency_converter tool."
        }}
    ]
}}
```

#### B. Example of Normal Conversation:
**User Request:** "Hello! I am John."

#### Tool Chain Logic:
1. Use the direct_answer tool to respond directly to the user.

**JSON Response Example:**
```json
{{
    "func_calling": [
        {{
            "tool_name": "direct_answer",
            "parameter": {{"query": "Hello! John, nice to meet you!"}},
            "call_ID": "1",
            "thought": "Responding directly to the user using the direct_answer tool."
        }}
    ]
}}
```

#### C. Example of Normal Conversation (llm_tool):
**User Request:** "How are you?"

#### Tool Chain Logic:
1. Use the llm_tool to respond to the user.

**JSON Response Example:**
```json
{{
    "func_calling": [
        {{
            "tool_name": "llm_tool",
            "parameter": {{"query": "How are you?"}},
            "call_ID": "1",
            "thought": "Generating text from the llm_tool."
        }}
    ]
}}
```

### Important Notes:
- **If a tool requires any parameter that is not available but is necessary to call, try to fill the parameter with a default value or at least don’t leave it blank.**
- All responses should strictly use only JSON and not plain text. If you want to reply directly to the user, use the direct_answer tool.
- **Utilize the direct_answer tool when you can answer the user’s question directly without using any other tools or performing tool chaining (mostly for conversation queries).**
- **Do not refer to specific previous turns or tool calls. The user sees only your current response.**
- **Do not hallucinate tool names, capabilities, or parameters. Use only the tools provided.**
- If a tool requires specific information, ensure it is available in the context/result of previous tool calls.
- If no tool is suitable, use the llm_tool.
- **Always guide or prompt the llm_tool responses from the user’s perspective and mostly give the user’s query directly to the llm_tool (if no changes are needed).**
- The llm_tool should be used to generate text responses only.
- **Ensure to fully analyze the user’s request and respond carefully, executing every task stated by the user in a single turn.**
- If a conversation query is uncertain, give the query as it is to the llm_tool.

### Query:
{self.objective}

### Conclusion:
Emphasize the significance of adhering to the outlined procedures to ensure seamless execution of tool chaining, error management, and successful handling of user requests. Implement different tool calls at once if needed.
''',
                messages=[],
            )
            response = self.llm.run("Generate JSON according to the objective.")
            self.llm.reset()

            action = self._parse_and_fix_json(response)
            print(f"{Fore.YELLOW}Response: {(action)}{Style.RESET_ALL}\n")
            if isinstance(action, str):  # Check if the response is correct
                return action

            results = {}
            for i, call in enumerate(action.get("func_calling", [])):
                tool_name = call["tool_name"]

                parameters = call.get("parameter", {})
                thought = call.get("thought", "")
                print(f"{Fore.MAGENTA}Thought: {thought}{Style.RESET_ALL}")
                for k, v in parameters.items():
                    if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                        for function in self.all_functions:
                            if function["function"]["name"] == tool_name:
                                break  # Stop after finding the desired tool
                        next_tool_call = self._get_next_tool_call(
                            results[str(i)], function, self.objective
                        )
                        if next_tool_call:
                            call["tool_name"] = next_tool_call["tool_name"]
                            call["parameter"] = next_tool_call["parameter"]

                try:
                    tool_response = self._call_tool(call, results)
                    results[call['call_ID']] = tool_response  
                    tool_response = {tool_name: tool_response}
                    print(f"{Fore.GREEN}{tool_response}{Style.RESET_ALL} ")
                except Exception as e:
                    print(f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}")
                    results[call['call_ID']] = f"Error: {e}"

            summary = self._generate_summary(results)

            # Logic to break the loop if the response is correct
            if self._is_response_valid(summary, action):  # Implement your validation logic
                print(f"{Fore.GREEN}Response is valid, stopping iterations.{Style.RESET_ALL}")
                return summary
            else:
                print(f"{Fore.YELLOW}Response is not valid, retrying...{Style.RESET_ALL}")

        # If max_iterations is reached and response is still invalid
        return summary  # Or handle the case differently (e.g., return an error)

    def _parse_and_fix_json(self, json_str: str) -> Dict | str:
        """Parses JSON string and attempts to fix common errors."""
        json_str = json_str.strip()
        if not json_str.startswith("{") or not json_str.endswith("}"):
            json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}JSON Error:{Style.RESET_ALL} {e}")

            json_str = json_str.replace("'", '"')
            json_str = re.sub(r",\s*}", "}", json_str) 
            json_str = re.sub(r"{\s*,", "{", json_str)
            json_str = re.sub(
                r"\s*,\s*", ",", json_str
            )  

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return f"Error: Could not parse JSON - {e}"

    def _direct_answer(self, query, tool_results):
        answer = query.get("query", "").lower()

        if answer:
            return f"Direct Answer: {answer}"

        return self._process_llm_tool(query, tool_results)

    def _call_tool(self, call, results):
        tool_name = call.get(
            "tool_name", ""
        ).lower() 
        query = call.get("parameter", {})

        if not tool_name:
            raise ValueError("No tool name provided in the call.")

        if tool_name == "direct_answer":
            return self._direct_answer(query, results)

        if tool_name == "llm_tool":
            return self._process_llm_tool(query, results)
        
        if tool_name == "ask_user":
            return self._ask_user(query)

        tool = next(
            (t for t in self.tools if t.func.__name__.lower() == tool_name),
            None,
        )
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        try:
            if tool.params:
                tool_response = tool.func(**query)
            else:
                tool_response = tool.func()
        except Exception as e:
            raise ValueError(f"Error executing tool '{tool_name}': {str(e)}")

        return tool_response if tool.returns_value else "Action completed."

    def _process_llm_tool(self, query, tool_results):
        if isinstance(query, dict):
            query = query.get("query", "")

        if isinstance(query, str):
            query = query.lower()
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
            You are {self.identity}, a highly capable AI assistant with the following attributes:
        - Core Purpose: {self.description}
        - Current Task: Respond to the user's query based on the provided context and previous tool outputs.
        - Output Style: {self.expected_output}

        Context:
        {context_str}

        User Query: {query}

        Key Instructions:
        1. Analyze the user's query and the provided context thoroughly.
        2. Generate a response that directly addresses the user's needs.
        3. Incorporate relevant information from the tool outputs when applicable.
        4. Maintain a coherent and logical flow in your response.
        5. If the query cannot be fully answered with the available information, clearly state what is known and what is uncertain.
        6. Do not fabricate information or claim access to data not provided in the context.
        7. Avoid referring to specific previous interactions or tool calls that the user cannot see.

        Remember: Your goal is to provide the most helpful and accurate response possible based on the given query and context.
            """,
            messages=[],
        )

        prompt_context = f"User Query:\n{query}\n\nTool Results:\n{context_str}"

        response = self.llm.run(prompt_context)
        self.llm.reset()
        return response

    def _generate_summary(self, results: Dict[str, str]) -> str:
        results_formatted = "\n".join(
            [f"Tool {call_id}: {output}" for call_id, output in results.items()]
        )
        prompt = f"User Query:\n{self.objective}\n\nTool Results:\n{results_formatted}\n\n"

        self.llm.__init__(
            system_prompt=f"""
            You are {self.identity}, an advanced AI agent with the following responsibilities:
        1. Synthesize information from various tool outputs to address the user's query comprehensively.
        2. Generate a coherent and informative response that aligns with the expected output style.

        Key Attributes:
        - Identity: {self.identity}
        - Description: {self.description}
        - Output Style: {self.expected_output}

        Guidelines:
        1. Carefully analyze all tool outputs provided.
        2. Construct a response that directly addresses the user's original query.
        3. Integrate relevant information from different tools seamlessly.
        4. Maintain factual accuracy - do not invent information not supported by tool outputs.
        5. If there are gaps in the information, acknowledge them clearly.
        6. Format your response for clarity and readability.
        7. Avoid referencing specific tool calls or previous interactions in your final response.

        Remember: Your goal is to provide a comprehensive, accurate, and helpful response that best serves the user's needs based on the available information.
        """,
            messages=[],
        )

        summary = self.llm.run(prompt)

        return summary

    def _get_next_tool_call(self, previous_tool_output: str, next_tool: Dict[str, Any], user_task: str) -> Dict[str, Any]:
        prompt = f"""
        Role: Expert Tool Chain Analyst

        Objective: Determine the optimal next tool in a processing chain based on:
        1. Previous tool output
        2. Available tools
        3. User's original task
        4. Overall context

        Context:
        - Original Task: {user_task}
        - Previous Output: {previous_tool_output}
        - Available Tool: {json.dumps(next_tool, indent=2)}

        Analysis Requirements:
        1. Evaluate tool compatibility with previous output
        2. Assess alignment with user's objective
        3. Consider data format requirements
        4. Identify potential processing gaps

        Decision Criteria:
        - Data compatibility
        - Task relevance
        - Processing efficiency
        - Expected outcome value

        Output Format:
        {{
            "tool_name": "<selected_tool>",
            "parameter": {{
                "<param_name>": "<processed_value>"
            }},
            "reasoning": "Brief explanation of tool selection and parameter processing"
        }}

        Additional Guidelines:
        - Ensure parameter values are properly formatted
        - Consider error handling requirements
        - Maintain data integrity across tool chain
        """
        
        self.llm.__init__(system_prompt="You are an expert Tool Chain Analyst specializing in optimal tool selection and parameter processing.")
        response = self.llm.run(prompt)
        self.llm.reset()
        
        try:
            parsed_response = self._parse_and_fix_json(response)
            if self.verbose and parsed_response.get("reasoning"):
                print(f"{Fore.CYAN}Tool Selection Reasoning: {parsed_response['reasoning']}{Style.RESET_ALL}")
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Error in tool selection: {e}{Style.RESET_ALL}")
            return {}
            
    def _is_response_valid(self, response: str, call: Dict[str, Any]) -> bool:
        if not self.check_validity:
            return True

        prompt = f"""
        Evaluate the validity of the given response considering the available tools and the current objective.

        Objective: {self.objective}
        Function Call: {json.dumps(call, indent=2)}
        Response: {response}
        Available Tools: {json.dumps(self.tool_info_for_validation, indent=2)}

        Evaluation Criteria:
        1. Accuracy: Does the response align with the information provided by the function call?
        2. Completeness: Does it address all aspects of the objective?
        3. Relevance: Is the information pertinent to the user's query?
        4. Consistency: Does it maintain logical coherence throughout?
        5. Tool Usage: Does it reflect appropriate use of the available tools?

        Guidelines:
        - Minor summarization or paraphrasing is acceptable if the core content is correct.
        - Slight omissions of non-critical details are permissible.
        - Focus on the overall quality and usefulness of the response.

        Return your evaluation in JSON format:
        {{
            "valid": <true or false>,
            "reason": "<brief explanation of your decision>"
        }}
        """

        self.llm.__init__(system_prompt="You are an AI Response Validator tasked with ensuring the quality and accuracy of AI-generated responses.")
        validation_result = self.llm.run(prompt)
        self.llm.reset()

        try:
            validation_result = self._parse_and_fix_json(validation_result)
            print(f"{Fore.YELLOW}Validation Reason: {validation_result.get('reason', '')}") 
            return validation_result.get("valid", False)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from validation LLM: {e}")
            return False  

    def _run(self) -> str:
        """Executes the agent's workflow, handling tool usage conditionally."""
        self.llm.reset()

        if self.tools:
            response = self._run_with_tools()
        else:
            response = self._run_no_tool()

        if self.file:
            with open(self.file, "a") as f:
                f.write(response)
        return response  

    def rollout(self) -> str:
        print(f"{Fore.LIGHTGREEN_EX}Executing {self.identity}...{Style.RESET_ALL}")

        response = self._run()  # Call the unified _run function
        with open(f"{self.cache_dir}/{self.identity}.md", "w") as f:
            f.write(response)
