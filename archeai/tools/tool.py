from typing import Callable, Dict, Any, Optional, Union, get_type_hints
import inspect
import json
from archeai.llms import Gemini  # Assuming this is your LLM class
from colorama import Fore, Style
import re

class Tool:
    def __init__(
        self,
        func: Callable,
        description: str,
        returns_value: bool = True,
        params: Optional[Dict] = None,
        instance: Optional[Any] = None,  # Class instance for method calls
        llm: Optional[Gemini] = None,
        verbose: bool = False,
    ):
        self.func = func
        self.instance = instance  # Optional class instance for bound methods
        self.name = func.__name__  # Automatically set the tool name
        self.description = description
        self.returns_value = returns_value
        self.llm = llm
        self.verbose = verbose
        if params:
            self.params = params
        else:
            self.params = self._extract_params()

    def _extract_params(self) -> Dict[str, Dict[str, Any]]:
        """Extracts parameter information, optionally using the LLM."""
        if self._has_params():
            if self.llm:
                return self._extract_params_with_llm()
            return self._extract_params_from_signature()
        if self.verbose:
            print(f"{Fore.YELLOW}No parameters found : {self.name}{Style.RESET_ALL}")
        return {}
    
    def _has_params(self) -> bool:
        """Check if the function has any parameters."""
        return len(inspect.signature(self.func).parameters) > 0

    def _parse_and_fix_json(self, json_str: str) -> Union[Dict[str, Any], str]:
        """Parses JSON string and attempts to fix common errors."""
        json_str = json_str.strip()
        if not json_str.startswith('{') or not json_str.endswith('}'):
            json_str = json_str[json_str.find('{'):json_str.rfind('}') + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"{Fore.RED}JSON Error:{Style.RESET_ALL} {e}")

            # Common JSON fixes
            json_str = json_str.replace("'", "\"")  # Replace single quotes
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r'{\s*,', '{', json_str)  # Remove leading commas
            json_str = re.sub(r'\s*,\s*', ',', json_str)  # Remove whitespaces around commas

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return f"Error: Could not parse JSON - {e}"

    def _extract_params_with_llm(self) -> Dict[str, Dict[str, Any]]:
        """Extracts parameters and descriptions using the LLM."""
        function_code = inspect.getsource(self.func).strip()
        prompt = f"""
        You are a Python expert. Given the following function:

        ```python
        {function_code}
        ```

        provide a JSON representation of its parameters and descriptions in the following format:

        ```json
        {{
          "param1": {{
            "description": "Description of param1",
            "type": "Optional[str]",
            "default": "default value or null" 
          }},
          "param2": {{
            "description": "Description of param2",
            "type": "...",
            "default": "..."
          }}
          // ... more parameters
        }}
        ```

        If the function has no parameters, return an empty JSON object: `{{}}`. 
        Make sure the JSON is valid. If the type or default value cannot be inferred, use 'unknown'.
        """

        try:
            response = self.llm.run(prompt).strip()
            params = self._parse_and_fix_json(response)
            if self.verbose:
                print(f"{Fore.GREEN}Initialzing {self.name}:{Style.RESET_ALL} {Fore.CYAN} {params}{Style.RESET_ALL}")
            return params  # Format and return params
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}Error parsing LLM response as JSON: {e}{Style.RESET_ALL}")
            if self.verbose:
                print(f"LLM Response: {response}")
            return self._extract_params_from_signature()  # Fallback to signature extraction

    def _extract_params_from_signature(self) -> Dict[str, Dict[str, Any]]:
        """Extracts parameters from the function signature."""
        params = {}
        hints = get_type_hints(self.func)  # Use get_type_hints for cleaner type extraction

        for name, param in inspect.signature(self.func).parameters.items():
            params[name] = {
                # "description": param.annotation.__doc__ if callable(param.annotation) else "No description provided.",
                "type": hints.get(name, "unknown").__name__ if hints.get(name) else "unknown",
                "default": param.default if param.default is not inspect.Parameter.empty else "unknown",
            }
        return params

    def format_json(self, data: Union[Dict[str, Any], str]) -> str:
        """Formats the given JSON data in a readable way."""
        if isinstance(data, str):
            try:
                data = json.loads(data)  # Parse string into JSON if necessary
            except json.JSONDecodeError as e:
                return f"Invalid JSON string: {e}"

        return json.dumps(data, indent=4, ensure_ascii=False)  # Pretty print with indentation

    def __call__(self, *args, **kwargs):
        """Makes the Tool object callable."""
        # Check if the tool is a bound method (i.e., has an instance)
        if self.instance:
            return self.func(self.instance, *args, **kwargs)  # Call with instance
        
        # Call the function directly if it has no parameters
        if not self._has_params():
            return self.func() 

        return self.func(*args, **kwargs)