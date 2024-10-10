from archeai import Tool
from archeai.llms import Gemini
from archeai.tools import web_search

hehe = Tool(func=web_search, description="hehe",returns_value=True, verbose=True)
print(hehe.params)

## Task : Get the current time and what should we do at this time and write the article in a file time.txt.