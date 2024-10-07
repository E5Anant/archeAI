try:
    from llms import Gemini
except ImportError:
    pass

import subprocess
import tempfile
import sys
import re
import json
from rich.syntax import Syntax
from prompts.codesmith import codesmithPrompt

from colorama import Fore, Back, Style
import colorama
colorama.init(autoreset=True)

def print_json(json_obj, indent=4):
    print(Fore.LIGHTGREEN_EX + "result: {")
    for key, value in json_obj.items():
        print(Fore.LIGHTGREEN_EX + ' ' * indent +Fore.LIGHTGREEN_EX +  f'{key} : ', end='')
        if isinstance(value, dict):
            print()
            print(Fore.LIGHTGREEN_EX + ' ' * indent + Fore.LIGHTGREEN_EX + "{")
            for nested_key, nested_value in value.items():
                print(' ' * (indent + 4) + Fore.LIGHTGREEN_EX +  f'{nested_key} : {nested_value},')
            print(Fore.LIGHTGREEN_EX + ' ' * indent +Fore.LIGHTGREEN_EX +  "},")
        else:
            print(Fore.LIGHTGREEN_EX + f'{value},')
    print(Fore.LIGHTGREEN_EX + "}")

def transform_gemini_messages(messages: list[dict[str, str]]):
    return [{"role": msg["role"].replace('assistant','model'), "parts": [msg["content"]]} for msg in messages]

class CodeSmith:
    def __init__(
            self,
            llm: Gemini,
            maxRetries: int = 3,
            keepHistory: bool = True,
            printScript: bool = True,
            printconfig: bool = True,
            ) -> None:
        self.llm: Gemini = llm
        self.maxRetries = maxRetries
        self.keepHistory = keepHistory
        self.verbose = printScript
        self.config = printconfig
        # print(self.llm)
        self.llm.__init__(system_prompt=codesmithPrompt())

    def filterCode(self, txt):
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, txt, re.DOTALL)
        for match in matches:
            return match.strip()
        return None
    
    def pipPackages(self, *packages: str):
        python_executable = sys.executable
        print(f"Installing {', '.join(packages)} with pip...")
        return subprocess.run(
            [python_executable, "-m", "pip", "install", *packages],
            capture_output=True,
            check=True,
        )
    
    def _execute_script_in_subprocess(self, script) -> tuple[str, str, int]:
        from rich import print
        output, error, return_code = "", "", 0
        try:
            python_executable = sys.executable
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_script:
                tmp_script_name = tmp_script.name
                tmp_script.write(script)
                tmp_script.flush()

                process = subprocess.Popen(
                    [python_executable, tmp_script_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,  # Raises EOF error if subprocess asks for input
                    text=True,
                )
                if self.verbose:print(Syntax(script, "python", theme="monokai"))
                while True:
                    _stdout = process.stdout.readline()
                    _stderr = process.stderr.readline()
                    if _stdout:
                        output += _stdout
                        print(_stdout, end="")
                    if _stderr:
                        error += _stderr
                        print(_stderr, end="", file=sys.stderr)
                    if _stdout == "" and _stderr == "" and process.poll() is not None:
                        break
                return_code = process.returncode
        except Exception as e:
            error += str(e)
            print(e)
            return_code = 1
        return output, error, return_code

    def execute_script(self, script: str) -> tuple[str, str, int]:
        return self._execute_script_in_subprocess(script)
    
    def run(self, prompt: str) -> None:
        if self.config:
            data = f'''
                            {{
                                "prompt": "{prompt}",
                                "keep_history": "{self.keepHistory}",
                                "max_retries": "{self.maxRetries}",
                                "commands": "press ctrl+c to cancel"
                            }}
                            '''
            data = json.loads(data)
            print_json(data)
        the_copy = self.llm.messages.copy()
        self.llm.add_message("user", prompt)
        _continue = True
        while _continue:
            _continue = False
            error, script, output, return_code = "", "", "", 0
            try:
                response = self.llm.run(prompt)
                if isinstance(self.llm, Gemini):
                    self.llm.add_message("model", response)
                else:
                    self.llm.add_message("assistant", response)
                script = self.filterCode(response)
                if script:
                    output, error, return_code = self.execute_script(script)
            except KeyboardInterrupt:
                break
            if output:
                self.llm.add_message("user", f"LAST SCRIPT OUTPUT:\n{output}")
                if output.strip().endswith("CONTINUE"):
                    _continue = True
            if error:
                self.llm.add_message("user", f"Error: {error}")
            if return_code != 0:
                self.maxRetries -= 1
                if self.maxRetries > 0:
                    print("Retrying...\n")
                    _continue = True
        if not self.keepHistory:
            self.llm.messages = the_copy
