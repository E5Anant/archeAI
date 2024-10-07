import json
import os

PROMPTS_DIR = fr"{os.path.join(os.getcwd(), 'prompts', 'codesmith', 'prompts')}"
def codesmithPrompt():
    with open(os.path.join(PROMPTS_DIR, "codesmith.jinja2"), "r") as f:
        return f.read()
    
if __name__=="__main__":
    print(codesmithPrompt())