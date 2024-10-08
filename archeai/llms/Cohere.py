import cohere 
import os
from dotenv import load_dotenv
from rich import print
from typing import Type,Optional

load_dotenv()

class Cohere:
    USER = "User"
    ASSISTANT = "assistant"
    SYSTEM = "System"
    def __init__(
            self,
            messages: list[dict[str, str]] = [],
            model: str = "command-r-plus",
            temperature:Optional[float] = 0.7,
            system_prompt:Optional[str] = None,
            max_tokens: int = 2048,
            connectors:Optional[list[str]] = [],
            verbose:Optional[bool] = False,
            api_key:str|None = None
            ) -> None:
        """
        Initialize the LLM

        Parameters
        ----------
        messages : list[dict[str, str]], optional
            The list of messages, by default []
        model : str, optional
            The model to use, by default "command-r-plus"
        temperature : float, optional
            The temperature to use, by default 0.0
        system_prompt : str, optional
            The system prompt to use, by default ""
        max_tokens : int, optional
            The max tokens to use, by default 2048
        connectors : list[str], optional
            The connectors to use, by default []
        verbose : bool, optional
            The verbose to use, by default False
        api_key : str|None, optional
            The api key to use, by default None

        Examples
        --------
        >>> llm = LLM()
        >>> llm.add_message("User", "Hello, how are you?")
        """
        self.api_key = api_key if api_key else os.getenv("COHERE_API_KEY")
        self.co = cohere.Client(api_key)
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.connectors = connectors
        self.verbose = verbose

        if self.system_prompt!=None:self.add_message(self.SYSTEM, self.system_prompt)

    def run(self, prompt: str) -> str:
        self.add_message(self.USER, prompt)
        """
        Run the LLM

        Parameters
        ----------
        prompt : str
            The prompt to run

        Returns
        -------
        str
            The response

        Examples
        --------
        >>> llm.run("Hello, how are you?")
        "I'm doing well, thank you!"
        """
        self.stream = self.co.chat_stream(
            model = self.model,
            message = prompt,
            temperature = self.temperature,
            chat_history = self.messages,
            connectors = self.connectors,
            preamble = self.system_prompt,
            max_tokens = self.max_tokens,
            )
        self.messages.pop()
        response:str = ""
        for event in self.stream:
            if event.event_type == "text-generation":
                if self.verbose:
                    print(event.text, end='')
                response += event.text
        return response

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the list of messages

        Parameters
        ----------
        role : str
            The role of the message
        content : str
            The content of the message

        Returns
        -------
        None

        Examples
        --------
        >>> llm.add_message("User", "Hello, how are you?")
        >>> llm.add_message("Chatbot", "I'm doing well, thank you!")
        """
        self.messages.append({"role": role, "message": content})
    
    def __getitem__(self, index) -> dict[str, str]|list[dict[str, str]]:
        """
        Get a message from the list of messages

        Parameters
        ----------
        index : int
            The index of the message to get

        Returns
        -------
        dict
            The message at the specified index

        Examples
        --------
        >>> llm[0]
        {'role': 'User', 'message': 'Hello, how are you?'}
        >>> llm[1]
        {'role': 'Chatbot', 'message': "I'm doing well, thank you!"}

        Raises
        ------
        TypeError
            If the index is not an integer or a slice
        """
        if isinstance(index, slice):
            return self.messages[index]
        elif isinstance(index, int):
            return self.messages[index]
        else:
            raise TypeError("Invalid argument type")

    def __setitem__(self, index, value) -> None:
        """
        Set a message in the list of messages

        Parameters
        ----------
        index : int
            The index of the message to set
        value : dict
            The new message

        Returns
        -------
        None

        Examples
        --------
        >>> llm[0] = {'role': 'User', 'message': 'Hello, how are you?'}
        >>> llm[1] = {'role': 'Chatbot', 'message': "I'm doing well, thank you!"}

        Raises
        ------
        TypeError
            If the index is not an integer or a slice
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
        self.stream = self.co.chat_stream(
            model = self.model,
            message = "",
            temperature = self.temperature,
            chat_history = self.messages,
            connectors = self.connectors,
            preamble = self.system_prompt,
            max_tokens = self.max_tokens,
            )

if __name__ == "__main__":
    llm = Cohere()
    llm.add_message("User", "Hello, how are you?")
    llm.add_message("Chatbot", "I'm doing well, thank you!")
    print(llm.run("write python code to make snake game"))