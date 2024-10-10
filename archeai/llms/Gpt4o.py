from dotenv import load_dotenv
from rich import print
import requests
import base64
import os

load_dotenv()

class Gpt4o:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    def __init__(
            self,
            messages: list[dict[str, str]] = [],
            model: str = "rohan/tune-gpt-4o",
            temperature: float = 0.0,
            system_prompt: str = "",
            max_tokens: int = 2048,
            verbose: bool = False,
            api_key: str | None = None
    ) -> None:
        """
        Initializes the LLM with the given parameters.

        Args
        ----
        messages: list[dict[str, str]]
            A list of messages to initialize the LLM with.
        model: str
            The model to use for the LLM.
        temperature: float
            The temperature to use for the LLM.
        system_prompt: str
            The system prompt to use for the LLM.
        max_tokens: int
            The maximum number of tokens to use for the LLM.
        verbose: bool
            Whether to print the response from the LLM.
        api_key: str | None
            The API key to use for the LLM.
        
        example:
        >>> llm = LLM()
        >>> llm.add_message("user", "Hello, how are you?")
        >>> llm.add_message("assistant", "I'm doing well, thank you!")
        >>> llm.run("Hello, how are you?")
        >>> "I'm doing well, thank you!"
        """
        self.api_key = api_key if api_key else os.environ["TUNE_STUDIO_API_KEY"]
        self.session =  requests.session()
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.verbose = verbose
        "" if not system_prompt else self.add_message(self.SYSTEM, system_prompt)
    def run(self, prompt: str|None = None) -> str:
        """
        Runs the LLM with the given prompt.

        Args
        ----
        prompt: str
            The prompt to use for the LLM.

        Returns
        -------
        str
            The response from the LLM.

        example:
        >>> llm = LLM()
        >>> llm.add_message("user", "Hello, how are you?")
        >>> llm.add_message("assistant", "I'm doing well, thank you!")
        >>> llm.run("Hello, how are you?")
        """

        "" if not prompt else self.add_message("user", prompt)
        url = "https://proxy.tune.app/chat/completions"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        self.data = {
        "temperature": self.temperature,
        
            "messages":  self.messages,
            "model": self.model,
            "stream": False,
            "frequency_penalty":  0.0,
            "max_tokens": self.max_tokens
        }
        "" if not prompt else self.messages.pop()
        response = self.session.post(url, headers=headers, json=self.data)
        print(response.json())
        return response.json()["choices"][0]["message"]["content"]

    def add_message(self, role: str, content: str, base64_image: str = "") -> None:
        """
        Adds a message to the LLM with the given role and content.

        Args
        ----
        role: str
            The role of the message.
        content: str
            The content of the message.
        base64_image: str
            The base64 image of the message.

        example
        -------
        >>> llm = LLM()
        >>> llm.add_message("user", "Hello, how are you?")
        >>> llm.add_message("assistant", "I'm doing well, thank you!")
        >>> llm.run("Hello, how are you?")
        >>> "I'm doing well, thank you!"
        """
        if content and base64_image:
            self.messages.append({
                "role": role,
                "content": [
                {
                    "type": "text",
                    "text": content
                },
                {
                    "type": "image_url", 
                    "image_url": 
                {
                    "url": f"data:image/png;base64,{base64_image}"
                }
                }
                ]
                })
        elif base64_image:
            self.messages.append(
                {
                "role": role, 
                "content": [
                    {
                    "type": "image_url", 
                    "image_url": 
                        {
                        "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]       
                }
                )
        elif content:
            self.messages.append(
                {
                "role": role, 
                "content": [
                    {
                    "type": "text",
                    "text": content
                }
                ]       
                }
                )
        else:
            raise ValueError("Both content and base64_image are None")

    
    def __getitem__(self, index) -> dict[str, str] | list[dict[str, str]]:
        """
        Returns the message at the given index.

        Args
        ----
        index: int
            The index of the message to return.

        Returns
        -------
        dict[str, str] | list[dict[str, str]]
            The message at the given index.

        example
        -------
        >>> llm = LLM()
        >>> llm.add_message("user", "Hello, how are you?")
        >>> llm.add_message("assistant", "I'm doing well, thank you!")
        >>> llm.run("Hello, how are you?")
        >>> "I'm doing well, thank you!"
        >>> llm[1]
        >>> llm[1:]
        """
        if isinstance(index, slice):
            return self.messages[index]
        elif isinstance(index, int):
            return self.messages[index]
        else:
            raise TypeError("Invalid argument type")

    def __setitem__(self, index, value) -> None:
        """
        Sets the message at the given index to the given value.

        Args
        ----
        index: int
            The index of the message to set.
        value: dict[str, str] | list[dict[str, str]]
            The value to set the message to.

        example
        -------
        >>> llm = LLM()
        >>> llm.add_message("user", "Hello, how are you?")
        >>> llm.add_message("assistant", "I'm doing well, thank you!")
        >>> llm.run("Hello, how are you?")
        >>> "I'm doing well, thank you!"
        >>> llm[1] = "I'm doing well, thank you!"
        """

        if isinstance(index, slice):
            self.messages[index] = value
        elif isinstance(index, int):
            self.messages[index] = value
        else:
            raise TypeError("Invalid argument type")
        
    def reset(self) -> None:
        """
        Reset the system prompts and messages

        Returns
        -------
        None
        """
        self.messages = []
        self.system_prompt = None
        self.data = {
        "temperature": self.temperature,
        
            "messages":  self.messages,
            "model": self.model,
            "stream": False,
            "frequency_penalty":  0.0,
            "max_tokens": self.max_tokens
        }

def FileToBase64(file_path:str):
    """
    Convert image file to base64 string.

    Args
    ----
    file_path : str

    Returns
    -------
    base64_image : str
    """
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


if __name__ == "__main__":
    llm = Gpt4o(verbose=True)
    llm.add_message("user", "Hello, how are you?")
    llm.add_message("assistant", "I'm doing well, thank you!")
    llm.run("Hello, how are you?")