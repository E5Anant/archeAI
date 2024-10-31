import os
from dotenv import load_dotenv
from typing import List, Dict
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

class Gemini:
    USER = "user"
    MODEL = "model"

    def __init__(self,
                 messages: list[dict[str, str]] = [],
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.0,
                 system_prompt: str|None = None,
                 max_tokens: int = 2048,
                 connectors: list[str] = [],
                 verbose: bool = False,
                 api_key: str|None = None
                 ):
        safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        self.api_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.connectors = connectors
        self.verbose = verbose
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
        )
        if self.system_prompt:
            self.add_message(self.MODEL, self.system_prompt)

    def run(self, prompt: str) -> str:
        self.add_message(self.USER, prompt)
        self.chat_session = self.client.start_chat(history=self.messages)
        response = self.chat_session.send_message(prompt)
        self.messages.pop()  # Remove the user prompt to avoid duplication
        r = response.text
        if self.verbose:
            print(r)
        return r

    def add_message(self, role: str, content: str) -> None:
        # Adjusting message structure for Gemini
        self.messages.append({"role": role, "parts": [content]})

    def __getitem__(self, index) -> dict[str, str]|list[dict[str, str]]:
        if isinstance(index, slice):
            return self.messages[index]
        elif isinstance(index, int):
            return self.messages[index]
        else:
            raise TypeError("Invalid argument type")

    def __setitem__(self, index, value) -> None:
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
        self.chat_session = self.client.start_chat(history=self.messages)


if __name__ == "__main__":
    llm = Gemini(system_prompt="your name is bob you like to tell jokes and memes")
    while True:
        q = input(">>> ")
        # llm.add_message(Gemini.USER, q)
        print(llm.run(q))
        # print("Before Reset:", llm.messages)
        # llm.reset()
        # print("After Reset:", llm.messages)

        
