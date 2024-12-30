import os
from archeai.agent import Agent
import shutil

def delete_directory_with_content(path):
    """Deletes a directory and all its contents.

    Args:
        path: The path to the directory to delete.
    """
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory '{path}': {e}")
    else:
        print(f"Directory '{path}' does not exist.")

class TaskForce:
    def __init__(self,
                 agents: list[Agent],
                 caching_dir:str = "cache",):

        self.caching_dir = caching_dir
        self.agents = agents

    def start_force(self):
        print("Force started")
        print(f"Cache directory: {self.caching_dir}")
        os.makedirs(self.caching_dir, exist_ok=True)
        for agent in self.agents:
            agent.cache_dir = self.caching_dir

    def execute_agent(self, agent:Agent, prompt:str):
        agent.objective = prompt
        agent.rollout()

    def record_result(self, agent:Agent):
        with open(f"{self.caching_dir}/{agent.identity}", "r") as f:
            result = f.read()
        return result
    
    def exit_force(self):
        print("Force exited")
        print("Deleting cache directory")
        delete_directory_with_content(self.caching_dir)
