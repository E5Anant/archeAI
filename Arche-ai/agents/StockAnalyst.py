from llms import Gemini
from typing import Type
import json
from tools import StockMarketInfo

class StockInfo:
    def __init__(self, llm: Type[Gemini]):
        self.llm = llm
        # Set the system prompt when initializing the LLM
        self.llm.__init__(system_prompt="""
        You are an AI agent designed to give ticker to search on the stock market. 
        You are incredible at accurate tickers. 

        Here's how to provide ticker for doing the Stock Market Info:

        **HOW TO PROVIDE TICKER:**
        Always respond with a JSON object with the following structure:
        {
            "calling": {
                "ticker": "ticker for searching the stock market eg tsla and aapl"
            }
        }
        ***EXAMPLE:***
        User: can I invest in "tsla"(your user may directly say the company name in that case, just give it's ticker name. or you can also use the name of the company.)
        
        You:
            {
            "calling": {
                "ticker": "tsla"
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
            query = action['calling']['ticker']
            info = StockMarketInfo()
            result = info.get_stock_details(query)
            return result

        except:
            return f"Failed to get info {Exception}."

class StockAnalyst:
    def __init__(self,
                 llm: Type[Gemini]) -> None:
        self.llm = llm
        self.data_agent = StockInfo(llm=self.llm) # Initialize data agent
        self.llm.__init__(system_prompt="""
        You are a Stock Analyst, an AI agent designed to interact with the stock market. 
        Your work is to summarize the stock details provided to you.
        
        ## Instructions:
        - Identify and prioritize the most important information from the stock results.
        - Provide a concise summary, aiming for around 100 words or less. 
        - Do not include any URLs or links in your response.
        - Only provide information that is directly present in the stock detail.
        
        ## Example:
        **Stock Results:** (Some lengthy stock search results)
        **Summary:** [summarized information in points - crisp and short
                          ***About Company***,
                          ***About Stocks***,
                          ***Upcoming Investments(if any)***,
                          ***Recent News***,
                          ***Concerns***,
                          ***Possibilities***,
                          ***Decision***].
        
        ***Remember: Your responses should be in text form only and not JSON or any other format.***""")

    def run(self, user_query: str) -> str:
        stock_results = self.data_agent.run(user_query)
        
        self.llm.add_message("user", stock_results)
        sweb_results = self.llm.run(f"***Summarize the stock results***\n{stock_results}")
        return sweb_results