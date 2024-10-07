from agents import WebSurfer
from llms import GroqLLM

agent = WebSurfer(llm=GroqLLM())

while True:
    print(agent.run(input(">>> ")))
