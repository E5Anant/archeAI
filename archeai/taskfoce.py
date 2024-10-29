from archeai.agent import Agent
import json
import re
from typing import List, Optional, Dict
import colorama
import os
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
        for agent in self.agents:
            agent.agents = self.agents
            self.objective = self.objective
            agent.passobjective = self.objective
            agent.mindmap = self.mindmap

        print(f"{Fore.GREEN}Mindmap:{Style.RESET_ALL}")
        print(self.mindmap)

    def _get_mindmap(self) -> str:
        self.llm.reset()
        self.llm.__init__(system_prompt=f"""
        You are managing a team of agents for task execution.
        You are tasked with generating a mindmap for the team.
        The team consists of {len(self.agents)} agents.

        # Team Members
        {self._get_agents_info()}
        
        # OBJECTIVE
        {self.objective}

        **Create a plan for the team of agents to complete the objective.**
        **Don't add unnecessary steps or agent calls which don't require to complete the objective.**
        **Try Getting the steps as minimum as possible**
        **The plan will be consisting of several steps to accomplish the objective.**
        **A step could contain the info for passing relevant info from one agent to another or which agent to execute with the appropirate tools.**
        **Try to get the plan as simple as possible, with minimum number of agent calls**""")
        result = self.llm.run("Create a plan for the team of agents to complete the objective")
        self.llm.reset()
        return result
    
    def rollout(self) -> str:
        if len(self.agents) == 1:
            for agent in self.agents:
                agent.objective = self.objective
                agent.passobjective = self.objective
                agent.agents = self.agents
            self.agents[0].rollout()
        else:
            self.execute()
        
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
        agent.rollout()

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
    
    def _get_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """Helper function to retrieve an agent by name."""
        return next((agent for agent in self.agents if agent.identity == agent_name), None)
