from archeai.agent import Agent
import json
import re
from typing import List, Optional, Dict
import colorama
from colorama import Fore, Style

colorama.init(autoreset=
    True
)

class TaskForce:
    def __init__(self,
                 agents: List[Agent],
                 objective:str = "No current task",
                 mindmap: Optional[str] = None) -> None:
        self.agents = agents
        self.llm = self.agents[0].llm
        self.objective = objective
        self.mindmap = mindmap
        if not self.mindmap:
            self.mindmap = self._get_mindmap()

        if len(self.agents) == 1:
            print("1 agent only")
            for agent in self.agents:
                agent.agents = self.agents
                agent.objective = self.mindmap
                agent.passobjective = self.objective
                agent.mindmap = self.mindmap
        else:
            for agent in self.agents:
                agent.agents = self.agents
                agent.objective = self.objective
                agent.passobjective = self.objective
                agent.mindmap = self.mindmap

        print(f"{Fore.GREEN}Mindmap:{Style.RESET_ALL}")
        print(self.mindmap)

    def _get_mindmap(self) -> str:
        self.llm.__init__(system_prompt=f"""
        ### Task:
Create a sequence for the team of agents to complete the specified objective efficiently. The sequence should include necessary tool calls, or passing relevant information between agents, without any unnecessary actions.

### Team Details:
The team consists of {len(self.agents)} agents.
{self._get_agents_info()}

### Objective:
{self.objective}

### Guidelines:
- Plan the sequence with the minimum number of steps required for accomplishing the objective.
- Include relevant info passing between agents or appropriate tool executions in each step.
- Avoid redundant steps, agent calls, or chat interactions that aren't needed to achieve the objective.
- For normal conversation objectives, aim to complete the task within a maximum of 2 steps.

### Example:
1. **Agent 1:**
   - Use Tool A to gather initial information (e.g., search, fetch data, etc.).
   - Pass the retrieved information to Agent 2.

2. **Agent 2:**
   - Process the information using Tool B.
   - Provide the final output to the user.

### Generate the Sequence
Please generate the sequence and steps based on the guidelines and the given objective. All agents are familiar with past conversations, so focus on building up the sequence rather than instructing them.
        """)
        if len(self.agents) == 1:
            result = self.objective
        else:
            result = self.llm.run("Create a sequence for the team of agents to complete the objective")
        self.llm.reset()
        return result
    
    def rollout(self) -> str:
        if len(self.agents) == 1:
            for agent in self.agents:
                agent.objective = self.mindmap
                agent.mindmap = self.mindmap
                agent.passobjective = self.objective
                agent.agents = self.agents
            return self.agents[0].rollout()
        else:
            return self.execute()
        
    def execute(self) -> str:
        agent_info = self._get_agents_info()
        self.llm.__init__(
            system_prompt=f"""
        You are managing a team of agents for task execution. 
        Your role is to assign tasks to agents, break down tasks if needed, and ensure effective communication.

        Current Task: {self.objective}
        Available Agents: 
        {agent_info} 

        # PLAN
        {self.mindmap}

        Instructions:
        1. Analyz the first step and choose the most suitable agent to execute the task.
        2. Return a JSON object:
        
        ```json
        {{
            "agent_name": "agent_name",
            "objective": "objective for the agent",
        }}"""
        )
        json_response = self.llm.run("Generate JSON for the first task")
        self.llm.reset()  # Reset LLM after the entire task rollout
        json_response = self._parse_and_fix_json(json_response)
        print(f"{Fore.GREEN}{json_response}{Style.RESET_ALL}")
        agent = self._get_agent_by_name(json_response["agent_name"])
        task = json_response["objective"]
        agent.objective = task
        agent.passobjective = self.objective  # Pass the overall objective to the agent
        return agent.rollout()

    def _parse_and_fix_json(self, json_str: str) -> Dict:
        """Parses JSON string and attempts to fix common errors. 
        Raises a ValueError if parsing is not possible.
        """
        json_str = json_str.strip()
        if not json_str.startswith("{") or not json_str.endswith("}"):
            json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Attempt to fix common JSON errors
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r"{\s*,", "{", json_str)
            json_str = re.sub(r"\s*,\s*", ",", json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If parsing still fails after fixes, raise a ValueError
                raise ValueError(f"Error: Could not parse JSON - {e}")

    def _get_agents_info(self) -> str:
        """Provides a formatted string of agent information for the LLM."""
        agent_info = []
        for agent in self.agents:
            tool_info = [tool for tool in agent.all_functions] if agent.all_functions else "Tools: None"
            current_task = agent.objective if agent.objective else "No current task"
            agent_info.append(
                f"Name: {agent.identity}\n"
                f"Description: {agent.description}\n"
                f"Tools :\n {tool_info}\n"
            )
        return "\n\n".join(agent_info)
    
    def _get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Helper function to retrieve an agent by name."""
        return next((agent for agent in self.agents if agent.identity == agent_name), None)