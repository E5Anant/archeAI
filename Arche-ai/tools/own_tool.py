from typing import Callable, Dict, Any, Optional
import inspect
import re
import json
from llms import Gemini
from colorama import Fore, Style

class Tool:
    def __init__(
        self,
        func: Callable,
        description: str,
        returns_value: bool = True,
        llm: Gemini = Optional[None],
        verbose: bool = False,
    ):
        self.ai = llm
        self.verbose = verbose
        self.func = func
        self.description = description
        self.returns_value = returns_value
        self.params = self._extract_params() 

    def _extract_params(self) -> Dict[str, Dict[str, Any]]:
        signature = inspect.signature(self.func)
        params = {}

        for name, param in signature.parameters.items():
            if param.annotation != inspect._empty:
                if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Optional:
                    param_type = param.annotation.__args__[0].__name__ 
                else:
                    param_type = param.annotation.__name__
            else:
                param_type = None

            if param_type is None and param.default != inspect._empty:
                param_type = type(param.default).__name__

            param_type = param_type or "string"

            if param.annotation != inspect._empty and hasattr(param.annotation, '__doc__'):
                description = param.annotation.__doc__.strip().split("\n")[0] if param.annotation.__doc__ else f"No description provided for '{name}'."
            else:
                description = f"No description provided for '{name}'."

            if param.default != inspect._empty:
                description += f" (default: {param.default})"

            params[name] = {
                "type": param_type,
                "description": self._get_enhanced_description(description, name),  
            }
        if self.verbose!=False:
            print(f"{Fore.LIGHTGREEN_EX} Initializing tool...{Style.RESET_ALL}{Fore.CYAN}'{self.func.__name__}'{Style.RESET_ALL}")
        return params

    def _get_enhanced_description(self, existing_description: str, param_name: str) -> str:

        if self.ai is None:
            return existing_description  

        # Construct the prompt for Julius (similar to your example)
        function_code = inspect.getsource(self.func)
        prompt = f"""Act as an expert Python Code Analyst. 

        The user will provide you with a Python function as a string. Your task is to analyze the function and extract the following information for a **single parameter**:

        1. Parameter Name: The name of the parameter being analyzed. (Already provided: '{param_name}')
        2. Description: 
            - If the function has a docstring or comments specifically describing the parameter's purpose, extract that description. 
            - If no specific description is provided for the parameter, analyze the function's code and generate a concise description of the parameter's likely purpose. If you cannot determine the purpose with reasonable confidence, set the description to "Purpose unclear. Further analysis needed."

        Based on the extracted information, create a dictionary in the following format:

        {{
          "parameter_name": "name_of_the_parameter",
          "description": "string_describing_the_parameter" 
        }}

        For example, if the user provides the following function and the parameter name is 'y':

        def add(x, y):
          \"\"\"
          This function adds two numbers together.
          \"\"\"
          return x + y

        Then your output should be:

        {{
          "parameter_name": "y",
          "description": "No specific description provided. Likely represents the second number to be added." 
        }}

        User: {function_code}

        Your Response: 
        """
        response = self.ai.run(prompt)

        # Extract the description from the JSON response
        try:
            response_json = self._parse_and_fix_json(response)
            if isinstance(response_json, dict) and "description" in response_json:
                enhanced_description = response_json["description"] 
            else:
                enhanced_description = existing_description
                print(f"Warning: Could not extract description from LLM response for '{param_name}'. Using existing description.")
        except Exception as e:  
            enhanced_description = existing_description
            print(f"Error processing LLM response for '{param_name}': {e}. Using existing description.")
        
        return enhanced_description


    def _parse_and_fix_json(self, json_str: str) -> Dict:
        """Parses JSON string and attempts to fix common errors."""

        json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Attempt to fix common JSON formatting errors
            json_str = json_str.replace("'", "\"")
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r'{\s*,', '{', json_str)
            json_str = re.sub(r'\s*,\s*', ',', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If parsing still fails, print the error and return an empty dict
                print(f"{Fore.RED}JSON Error:{Style.RESET_ALL} {e}")
                return {}  

if __name__ == "__main__":
    from web_search import web_search
    ai = Gemini() 
    boom = Tool(func=web_search, description="", returns_value=True, ai=ai)
    print(boom.params)