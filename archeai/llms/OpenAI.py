from openai import OpenAI
import os
from dotenv import load_dotenv
from rich import print
from typing import Type, Optional

load_dotenv()

class Openai:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    def __init__(
            self,
            messages: list[dict[str, str]] = [],
            model: str = "gpt-4-1106-preview",  # Updated default model
            temperature: Optional[float] = 0.7,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            api_key: str | None = None
    ) -> None:
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        if self.system_prompt is not None:
            self.add_message(self.SYSTEM, self.system_prompt)

    def run(self, prompt: str) -> str:
        self.add_message(self.USER, prompt)
        response_content = ""
        self.stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": message["role"], "content": message["message"]}
                for message in self.messages
            ],
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages.pop()
        for chunk in self.stream:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")
        return response_content

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "message": content})

    def __getitem__(self, index) -> dict[str, str] | list[dict[str, str]]:
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
        self.messages = []
        self.system_prompt = None
        self.stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": message["role"], "content": message["message"]}
                for message in self.messages
            ],
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

if __name__ == "__main__":
    llm = Openai(model="gpt-3.5-turbo") 
    llm.add_message("User", "Hello, how are you?")
    llm.add_message("Chatbot", "I'm doing well, thank you!")
    print(llm.run("Say this is a test"))