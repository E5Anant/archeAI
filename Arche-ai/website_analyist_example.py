from agents import WEBAnalyst
from llms import Gemini

print(WEBAnalyst(url="https://icrisstudio1.pythonanywhere.com/", llm=Gemini()).run())