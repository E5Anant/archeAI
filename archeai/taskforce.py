import json
from typing import List, Dict, Optional, Tuple, Any
from archeai.llms import Gemini  # Your LLM class
from archeai import Agent  # Your Agent class (remains unchanged)
from colorama import Fore, Style

class TaskForce:
    def __init__(self, agents: List[Agent], llm: Gemini, name: str, description: str, verbose: bool = False):
        self.agents = agents
        self.llm = llm
        self.name = name
        self.description = description
        self.verbose = verbose
        self.task_history: List[Tuple[str, str, str]] = []  # Track agent, task, and response
        self.shared_workspace: Dict[str, Any] = {}  # Shared workspace for general data
        self.message_broker = MessageBroker()  # Initialize the message broker

    def add_agent(self, agent: Agent):
        """Adds an agent to the task force."""
        self.agents.append(agent)

    def remove_agent(self, agent_name: str):
        """Removes an agent from the task force by name."""
        self.agents = [agent for agent in self.agents if agent.name != agent_name]

    def rollout(self, initial_task: str, max_iterations: int = 5) -> str:
        """Orchestrates task execution and agent collaboration."""

        current_task = initial_task
        self.task_history.clear()
        self.shared_workspace.clear()

        print(f"{Fore.CYAN}TaskForce activated. Initial task: {initial_task}{Style.RESET_ALL}")

        for iteration in range(max_iterations):
            print(f"\n{Fore.CYAN}Iteration: {iteration + 1}/{max_iterations}{Style.RESET_ALL}")

            selected_agent, next_task, communication_plan = self._plan_iteration(current_task)

            if not selected_agent:
                print(f"{Fore.YELLOW}Task planning complete or no suitable agent found.{Style.RESET_ALL}")
                break

            print(f"{Fore.YELLOW}Selected Agent: {selected_agent.name}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Task: {current_task}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Communication Plan: {communication_plan}{Style.RESET_ALL}\n")

            # Inject messages into the next task description
            if communication_plan:
                next_task = self._inject_messages_into_task(next_task, communication_plan)

            # Process and log communication via the message broker
            self.message_broker.process_communications(communication_plan)

            print(f"{Fore.GREEN}Executing Agent: {selected_agent.name}...{Style.RESET_ALL}")

            # Delegate the task (now potentially containing messages) to the agent
            selected_agent.task = next_task
            agent_response = selected_agent.rollout()

            print(f"{Fore.GREEN}Agent {selected_agent.name} Response: {agent_response}{Style.RESET_ALL}")

            self.shared_workspace[selected_agent.name] = agent_response
            self.task_history.append((selected_agent.name, next_task, agent_response))

            if next_task.upper() == "TASK COMPLETE":
                print(f"{Fore.GREEN}Task successfully completed!{Style.RESET_ALL}")
                break

            current_task = next_task

        final_response = self._generate_final_response(initial_task)

        if iteration + 1 == max_iterations:
            print(f"{Fore.YELLOW}Maximum iterations reached. Task might not be fully complete.{Style.RESET_ALL}")

        print(f"{Fore.CYAN}\nFinal Consolidated Response:{Style.RESET_ALL}\n{final_response}")
        return final_response

    def _plan_iteration(self, current_task: str) -> Tuple[Optional[Agent], str, dict]:
        """Plans the next iteration, handling task delegation and communication."""

        agent_info = self._get_agents_info()

        llm_prompt = f"""
        You are managing a team of agents for task execution. 
        Your role is to assign tasks to agents, break down tasks if needed, and ensure effective communication.

        Current Task: {current_task}
        Shared Workspace: {json.dumps(self.shared_workspace, indent=4)}
        Available Agents: 
        {agent_info}

        Instructions:
        1. Choose the most suitable agent for the next task. If none, suggest "None".
        2. If the task can be broken into smaller sub-tasks, suggest the next task.
        3. Plan communication between agents if necessary.
        4. Return a JSON object:
        {{
            "selected_agent": "<Agent Name>" or "None",
            "next_task": "<Task Description>" or "TASK COMPLETE",
            "communication_plan": {{
                "<Recipient Agent Name>": {{
                    "message": "<Information to send>",
                    "source_agent": "<Source Agent Name (if applicable)>",
                    "priority": "<low, medium, high>"
                }}
            }}
        }}
        """

        response = self.llm.run(llm_prompt)
        self.llm.reset()

        if self.verbose:
            print(f"LLM Planning Response: {response}")

        plan = self._extract_json_plan(response)
        if not plan:
            print(f"{Fore.RED}Error: Invalid plan format from LLM.{Style.RESET_ALL}")
            return None, "TASK COMPLETE", {}

        selected_agent_name = plan.get("selected_agent")
        next_task = plan.get("next_task", "TASK COMPLETE")
        communication_plan = plan.get("communication_plan", {})

        selected_agent = self._get_agent_by_name(selected_agent_name) if selected_agent_name != "None" else None
        return selected_agent, next_task, communication_plan

    def _inject_messages_into_task(self, task: str, communication_plan: Dict[str, dict]) -> str:
        """Injects messages from the communication plan into the task description."""
        for recipient_name, comm_details in communication_plan.items():
            recipient_agent = self._get_agent_by_name(recipient_name)  # Get the agent
            if recipient_agent and recipient_name == recipient_agent.name:  # Check if agent exists
                message = comm_details["message"]
                source_agent = comm_details["source_agent"]
                task += f"\n\n**Message from {source_agent}:** {message}"
        return task

    def _generate_final_response(self, initial_task: str) -> str:
        """Combines agent responses into a final answer."""

        final_response_prompt = f"""
        You are {self.name}, {self.description}.
        Combine the agents' responses to provide a comprehensive answer to the initial task.

        Initial Task: {initial_task}
        Shared Workspace: {json.dumps(self.shared_workspace, indent=4)}
        Task History: 
        {self._format_task_history()}

        Instructions:
        - Synthesize the information from the shared workspace and task history.
        - Provide a concise, unified answer to the initial task.
        """

        final_response = self.llm.run(final_response_prompt)
        self.llm.reset()

        if self.verbose:
            print("Final Response:")
            print(final_response)

        return final_response

    def _get_agents_info(self) -> str:
        """Provides a formatted string of agent information for the LLM."""
        agent_info = []
        for agent in self.agents:
            tool_names = [tool.func.__name__ for tool in agent.tools] if agent.tools else []
            tool_info = f"Tools: {', '.join(tool_names)}" if tool_names else "Tools: None"
            current_task = agent.task if agent.task else "No current task"
            agent_info.append(
                f"Name: {agent.name}\n"
                f"Description: {agent.description}\n"
                f"Skills: {agent.skills}\n"
                f"{tool_info}\n"
                f"Current Task: {current_task}"
            )
        return "\n\n".join(agent_info)

    def _format_task_history(self) -> str:
        """Formats the task history for the LLM."""
        if not self.task_history:
            return "Task History: None"
        history_str = ["Task History:"]
        for i, (agent_name, task, response) in enumerate(self.task_history):
            history_str.append(
                f"- Turn {i + 1}: Agent '{agent_name}' was given the task '{task}' and responded with '{response}'"
            )
        return "\n".join(history_str)

    def _extract_json_plan(self, llm_response: str) -> Optional[Dict]:
        """Extracts and validates the JSON plan from the LLM's response."""
        try:
            llm_response = llm_response.strip()
            start = llm_response.index('{')
            end = llm_response.rindex('}') + 1
            llm_response = llm_response[start:end]
            plan = json.loads(llm_response)
            assert "selected_agent" in plan and "next_task" in plan
            return plan
        except (json.JSONDecodeError, AssertionError, ValueError) as e:
            print(f"{Fore.RED}Error parsing JSON: {e}{Style.RESET_ALL}")
            return None

    def _get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Helper function to retrieve an agent by name."""
        return next((agent for agent in self.agents if agent.name == agent_name), None)


class MessageBroker:
    """Centralized message broker for managing inter-agent communication."""

    def __init__(self):
        self.message_log: List[Dict[str, Any]] = []  # Logs all communications

    def process_communications(self, communication_plan: Dict[str, dict]):
        """Processes communications, logging each message."""
        for recipient_name, comm_details in communication_plan.items():
            message = comm_details["message"]
            source_agent = comm_details["source_agent"]
            priority = comm_details.get("priority", "medium")

            print(
                f"{Fore.MAGENTA}Message from {source_agent} to {recipient_name}: '{message}' "
                f"(Priority: {priority}) - Logged by MessageBroker{Style.RESET_ALL}"
            )

            self.message_log.append({
                "recipient": recipient_name,
                "source": source_agent,
                "message": message,
                "priority": priority
            })