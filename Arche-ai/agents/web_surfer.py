import json
from llms import GroqLLM
from typing import Type
from tools import web_search

class DATA_:
    def __init__(self, llm: Type[GroqLLM]):
        self.llm = llm
        # Set the system prompt when initializing the LLM
        self.llm.__init__(system_prompt="""
        You are an AI agent designed to give queries to search on the web. 
        You are incredible at designing a crisp and accurate query. 

        Here's how to provide query for doing the web_search:

        **HOW TO PROVIDE QUERY:**
        Always respond with a JSON object with the following structure:
        {
            "calling": {
                "query": "query for searching the web"
            }
        }

        ***Always respond in a concise manner. Always return a JSON object as described above. ***""")

    def run(self, user_query: str) -> str:
        """
        Executes the main loop of the WebSurfer agent.
        """
        response = self.llm.run(user_query)
        
        # Ensure we strip any extraneous whitespace and newlines
        response = response.strip()

        # print("Raw response from LLM:", response)  # Debugging: print raw response

        try:
            # Ensure the response is valid JSON
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            action = json.loads(response)
            query = action['calling']['query']

            # print("Query for web_search:", query)  # Debugging: print the query

            web_results = web_search(query)
            # print("Web search results:", web_results)  # Debugging: print the web search results

            # Provide context and ask for a summary WITHOUT JSON format
            return web_results

        except Exception as e:
            return f"Your query could not be processed: {e}"
        
class WebSurfer:
    def __init__(self,
                 llm: Type[GroqLLM]) -> None:
        self.llm = llm
        self.data_agent = DATA_(llm=self.llm) # Initialize data agent
        self.llm.__init__(system_prompt="""
        You are a WebSurfer, an AI agent designed to interact with the web. 
        Your work is to summarize the web results provided to you.
        
        ## Instructions:
        - Identify and prioritize the most important information from the web results.
        - Provide a concise summary, aiming for around 100 words or less. 
        - Do not include any URLs or links in your response.
        - Only provide information that is directly present in the web results.
        
        ## Example:
        **Web Results:** (Some lengthy web search results)
        **Summary:** [summarized information in crisp and short].
        
        ***Remember: Your responses should be in text form only and not JSON or any other format.***""")

    def run(self, user_query: str) -> str:
        web_results = self.data_agent.run(user_query)
        
        self.llm.add_message("user", web_results)
        sweb_results = self.llm.run(f"***Summarize the web results***\n{web_results}")
        return sweb_results


if __name__ == "__main__":
    # Make sure the Gemini class is correctly instantiated and accessible
    llm = GroqLLM(verbose=True)
    web_surfer = WebSurfer(llm=llm)

    while True:
        user_query = input("You: ")
        response = web_surfer.run(user_query)
        print(f"WebSurfer: {response}")
