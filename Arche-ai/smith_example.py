from plugins import CodeSmith
from llms import Gemini

def transform_messages(messages: list[dict[str, str]]):
        return [{"role": msg["role"].replace('assistant','model'), "parts": msg["content"]} for msg in messages]


llm = Gemini(verbose=True, max_tokens=4096)
smith = CodeSmith(llm, keepHistory=True)

while 1:
    smith.run(input(">>> "))