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
        identity: str = "Agent",
        description: str = "A helpful AI agent.",
        expected_output: str = "Concise and informative text.",
        objective: str = "Hello, how are you?",
        verbose: bool = False,
        memory: bool = True,
        memory_dir: str = "memories",
        max_memory_responses: int = 12,
        max_iterations: int = 3,  # Add max_iterations parameter
    ) -> None:

        self.llm = llm
        self.tools = tools
        self.identity = identity
        self.description = description
        self.expected_output = expected_output
        self.objective = objective
        self.verbose = verbose
        self.passobjective = False
        self.agents = None
        self.memory_enabled = memory
        self.max_iterations = max_iterations  # Assign to the instance variable

        self.memory_dir = memory_dir
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.memory = Memory(
            llm=self.llm,
            status=self.memory_enabled,
            assistant_name=self.identity,
            history_dir=self.memory_dir,
            max_responses=max_memory_responses,
            update_file=self.memory_enabled,
            memory_filepath=f"{self.identity}_memory.txt",
            chat_filepath=f"{self.identity}_chat.txt",
            system_prompt=f"You are {self.identity}, {self.description}.",
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
                "direct_answer",
                "Use this tool when you can directly answer the user's query from the context or your knowledge. Avoid using this tool if the user's query requires specific actions or if you need to gather external information.",
                answer={
                    "type": "str",
                    "description": "The direct answer to the user's query.",
                },
            ),
        ]
        self.tool_info_for_validation = [
            {"name": tool.func.__name__, "description": tool.description} for tool in self.tools
        ]

    def add_tool(self, tool: Tool):
        """Add a tool dynamically."""
        self.tools.append(tool)
        self.all_functions.append(
            convert_function(
                tool.func.__name__, tool.description, **(tool.params or {})
            )
        )
        self.tool_info_for_validation.append({"name": tool.func.__name__, "description": tool.description})

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
        self.tool_info_for_validation = [
            info for info in self.tool_info_for_validation if info["name"] != tool_name
        ]

    def _run_no_tool(self) -> str:
        for current_iteration in range(self.max_iterations):  # Iteration loop
            if self.verbose:
                print(f"{Fore.CYAN}Iteration: {current_iteration + 1}{Style.RESET_ALL}")
            """Handles user queries when no tools are available."""
            prompt = self.memory.gen_complete_prompt(self.objective)
            self.llm.__init__(
                system_prompt=f"""
            **You are {self.identity}, {self.description}.**

            ### OBJECTIVE:
            {self.objective}

            ### OUTPUT STYLE:
            {self.expected_output}

            """,
                messages=[],
            )
            result = self.llm.run(prompt)
            self.llm.reset()
            self.memory.update_chat_history(self.identity, result, force=True)
            if self._is_response_valid(result, {"response": result}):  # Implement your validation logic
                if self.verbose:
                    print(f"{Fore.GREEN}Response is valid, stopping iterations.{Style.RESET_ALL}")
                print("Response:")
                print(result)
                return result
            else:
                if self.verbose:
                    print(f"{Fore.YELLOW}Response is not valid, retrying...{Style.RESET_ALL}")
        print("Response:")
        print(result)
        return result

    def _agent_pass(self, response) -> str:
        """Passes information to other agents if necessary."""
        agents = self._get_agents_info()
        self.llm.__init__(
            system_prompt = f"""
            As the Communication Handler, evaluate whether to pass the information to another agent or end the conversation based on the current response and overall objective.

            # AVAILABLE AGENTS:
            {agents}

            # Previous Agent Interaction:
            {{{self.identity} : {response}}}

            # Previous Response and Details:
            **Agent** - {self.identity}
            **Description** - {self.description}
            **Response** - {response}

            You are tasked to decide whether to:

            PASS the information to the next most suitable agent.
            END the conversation and return the current response as the final answer.
            REASONING:
            Carefully consider the current ‘response’, the overall ‘objective,’ and previous agent interactions.

            If the ‘response’ seems like a complete and satisfactory answer to the ‘objective’, choose END.
            If a sub-task within the ‘objective’ still needs to be addressed, and another agent is more suitable, choose PASS.
            # OUTPUT FORMAT:
            Only reply in one of these two JSON formats:

            # To PASS to another agent:

            ```json
            {{
            "decision": "PASS",
            "agent": "<agent_name>",
            "prompt": "Prompt for the agent (Markdown format, highlight important info with **)",
            "thought": "Summary of the decision process (previous agent's actions, reason for passing)"
            }}
            ```
            # To END the conversation:

            ```JSON
            {{
            "decision": "END"
            }}
            ```
            **Include tool use inside the prompt when applicable.**

            # OBJECTIVE:
            {self.passobjective}
        
        """)

        pass_info = self.llm.run("Generate only JSON")
        self.llm.reset()
        pass_info = self._parse_and_fix_json(pass_info)

        if pass_info.get("decision") == "PASS":
            if self.verbose:
                print(f"{Fore.CYAN}{pass_info}{Style.RESET_ALL}")
            agent_name = pass_info.get("agent")
            if agent_name:
                agent = self._get_agent_by_name(agent_name)
                if agent:
                    agent.objective = pass_info.get("prompt")
                    print(f"Passing to agent: {agent_name}")
                    return agent.rollout() 

        # If "decision" is not "PASS", return the current response
        print(f"{Fore.BLUE}Final Consolidated response:{Style.RESET_ALL}")
        print(response)
        return response 

    def _get_agent_by_name(self, agent_name: str):
        """Helper function to retrieve an agent by name."""
        return next((agent for agent in self.agents if agent.identity == agent_name), None)

    def _get_agents_info(self) -> str:
        """Provides a formatted string of agent information for the LLM."""
        agent_info = []
        for agent in self.agents:
            tool_names = [tool.func.__name__ for tool in agent.tools] if agent.tools else []
            tool_info = f"Tools: {', '.join(tool_names)}" if tool_names else "Tools: None"
            current_task = agent.objective if agent.objective else "No current task"
            agent_info.append(
                f"Name: {agent.identity}\n"
                f"Description: {agent.description}\n"
                f"{tool_info}\n"
                f"Current Task: {current_task}"
            )
        return "\n\n".join(agent_info)

    def _run_with_tools(self) -> str:
        for current_iteration in range(self.max_iterations):  # Iteration loop
            if self.verbose:
                print(f"{Fore.CYAN}Iteration: {current_iteration + 1}{Style.RESET_ALL}")

            # Initializing LLM with an improved system prompt
            self.llm.__init__(
                system_prompt=f"""
                **You are the json instructor of the AI agent `{self.identity}`, an advanced AI assistant capable of using multiple tools in sequence (tool chaining) to fulfill complex user requests. Your task is to manage the execution of tools, process their outputs, and pass them as inputs to subsequent tools when necessary.
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
               "call_ID": "",
               "thought": "a brief description of the tool call or the performed action."
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
               "call_ID": "1",
               "thought": "Searching the web for current pitcoin price from web_search tool."
           }},
           {{
               "tool_name": "currency_converter",
               "parameter": {{"amount": "{{1.output}}", "from": "BTC", "to": "USD"}},
               "call_ID": "2",
               "thought": "Converting Bitcoin price to USD from currency_converter tool."
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
               "call_ID": "1",
               "thought": "Generating text from llm_tool tool."
           }}
       ]
   }}


#### VII. Important Notes
- Only the `llm_tool` should be used for normal conversations.
- All responses should strictly use only JSON and not plain text and if want to reply directly to the user adher to `direct_answer` tool.
- Use the ‘direct_answer’ tool when you can directly answer the user’s question without using any other tools nor performing tool chaining (mostly for conversation queries). This would return directly your response to the user.
- **Do not refer to specific previous turns or tool calls.** The user only sees your current response.
- Do not hallucinate tool names or capabilities. You can only use the tools provided. 
- If a tool requires specific information, make sure it is available in the context or from previous tool calls. 
- If no tool is suitable, use the 'llm_tool' for a conversational response.
- Avoid generating text and always guide the `llm_tool` responses from the user's perspective.
- The `llm_tool` should be used to generate text responses only.
- **Always reply in JSON.**
- **Make sure to fully analyze the user's request and respond carefully, execute every task said by the user at a single turn.**
- **Analyze the user's query and respond accordingly.**
- If a conversation query is doubtfull to you, give the query as is to the `llm_tool`.

# OBJECTIVE:
{self.objective}

# Conclusion
Emphasize the significance of adhering to the outlined procedures to ensure the seamless execution of tool chaining, error management, and the successful handling of user requests. Adhere to different tool calls at once if needed.
                """,
                messages=[],
            )

            if self.memory_enabled:
                prompt = self.memory.gen_complete_prompt(self.objective)
                response = self.llm.run(prompt)
            else:
                response = self.llm.run("Generate JSON according to the objective.")
            self.llm.reset()

            action = self._parse_and_fix_json(response)
            if self.verbose:
                print(f"{Fore.YELLOW}Response: {(action)}{Style.RESET_ALL}\n")
            if isinstance(action, str):  # Check if the response is correct
                return action

            results = {}
            for i, call in enumerate(action.get("func_calling", [])):
                tool_name = call["tool_name"]

                parameters = call.get("parameter", {})
                thought = call.get("thought", "")
                if self.verbose:
                    print(f"{Fore.MAGENTA}Action: {thought}{Style.RESET_ALL}")
                for k, v in parameters.items():
                    if isinstance(v, str) and re.search(r"{\d+\.output}", v):
                        for function in self.all_functions:
                            if function["function"]["name"] == tool_name:
                                func = function
                                break  # Stop after finding the desired tool
                        next_tool_call = self._get_next_tool_call(
                            results[str(i)], func, self.objective
                        )
                        if next_tool_call:
                            call["tool_name"] = next_tool_call["tool_name"]
                            call["parameter"] = next_tool_call["parameter"]

                try:
                    tool_response = self._call_tool(call, results)
                    results[call['call_ID']] = tool_response  
                    if self.verbose:
                        tool_response = {tool_name: tool_response}
                        print(f"{Fore.GREEN}{tool_response}{Style.RESET_ALL} ")
                except Exception as e:
                    if self.verbose:
                        print(f"{Fore.RED}Tool Error ({call['tool_name']}):{Style.RESET_ALL} {e}")
                    results[call['call_ID']] = f"Error: {e}"

            if self.memory_enabled:
                self.memory.update_chat_history("Tools", json.dumps(results, indent=2), force=True)

            summary = self._generate_summary(results)

            # Logic to break the loop if the response is correct
            if self._is_response_valid(summary, action):  # Implement your validation logic
                if self.verbose:
                    print(f"{Fore.GREEN}Response is valid, stopping iterations.{Style.RESET_ALL}")
                return summary
            else:
                if self.verbose:
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
            if self.verbose:
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
            **You are {self.identity}, {self.description}.**
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
            else query
        )

        response = self.llm.run(prompt_context)
        self.llm.reset()
        return response

    def _generate_summary(self, results: Dict[str, str]) -> str:
        results_formatted = "\n".join(
            [f"Tool {call_id}: {output}" for call_id, output in results.items()]
        )
        prompt = f"User Query:\n{self.objective}\n\nTool Results:\n{results_formatted}\n\n"

        if self.memory_enabled:
            prompt = self.memory.gen_complete_prompt(prompt)

        self.llm.__init__(
            system_prompt=f"""
            You are {self.identity}, an AI agent. {self.description}.
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
            self.memory.update_chat_history(self.identity, summary, force=True)

        return summary

    def _get_next_tool_call(self, previous_tool_output: str, all_tools, user_task: str) -> dict:
        prompt = f"""
        You are an AI assistant helping to choose the best tool to use for a given task. 

        The user's original task is: {user_task}

        The output from the previous tool call is: {previous_tool_output}

        Available tools:
        {all_tools}

        Based on the user's task and the previous tool's output, determine the most logical next tool to call. 
        If no tool is suitable, return an empty JSON object. 

        Output your suggestion in the following JSON format:
        {{
            "tool_name": "<tool_name>",
            "parameter": {{<param_name>: "<param_value>"}} 
        }}

        For example:
        {{
            "tool_name": "weather_tool",
            "parameter": {{"location": "London"}}
        }}
        """

        self.llm.__init__(system_prompt="You are an assistant helping to choose the best tool...")
        response = self.llm.run(prompt)
        self.llm.reset()

        if self.verbose:
            print("Intermediary LLM response:")
            print(response)

        try:
            response = self._parse_and_fix_json(response)
            return response
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from intermediary LLM: {e}")
            return {}

    def _is_response_valid(self, response: str, call) -> bool:
        """Uses the LLM to evaluate the response validity based on the current objective."""
        prompt = f"""
        Evaluate the validity of the given response considering the available tools.

        **Your Objective**: {self.objective}
        **Function Call**: {call}
        **Response**: {response}
        **Available Tools**: {self.tool_info_for_validation}

        Based on the current objective, available tools, and the function call, is the response valid?

        * Check if the claims made in the response are true, as per the ‘Function Call’.
        * Even if the response does not show the content, ensure it aligns with the ‘Function Call’.
        * Ignore any minor summarization errors.
        * If minor ignorable parts are removed from the response, consider it as valid.
        * If the response lacks minor details, it is still valid if the core content is correct.
        * Minor errors in the response should be ignored.

        Return your answer in JSON format:
        {{
            "valid": <true or false>,
            "reason": "<brief explanation>" 
        }}
        """

        self.llm.__init__(system_prompt="You are an AI assistant tasked with evaluating the validity of a response.")
        validation_result = self.llm.run(prompt)
        self.llm.reset()

        try:
            validation_result = self._parse_and_fix_json(validation_result)
            if self.verbose:
                print(f"Validation Result: {validation_result}") 
            return validation_result.get("valid", False) 
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from validation LLM: {e}")
            return False  

    def _run(self) -> str:
        """Executes the agent's workflow, handling tool usage conditionally."""
        self.memory.update_chat_history("User", self.objective, force=True)
        self.llm.reset()

        if self.tools:
            response = self._run_with_tools()
        else:
            response = self._run_no_tool()

        return response  

    def rollout(self) -> str:
        if self.verbose:
            print(f"{Fore.LIGHTGREEN_EX}Executing {self.identity}...{Style.RESET_ALL}")

        response = self._run()  # Call the unified _run function
        self._agent_pass(response)
