from agents import StockAnalyst
from llms import GroqLLM

llm = GroqLLM()
agent = StockAnalyst(llm=llm)

while True:
    print(agent.run(input(">>> ")))