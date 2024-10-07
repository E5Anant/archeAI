from llms import Gemini  # Assuming you have defined this for your LLM
from tools import HTMLContentScraper # Assuming this is your previous class
from typing import Type

class WEBAnalyst:
    def __init__(self, url: str, llm: Type[Gemini]):
        self.url = url
        self.llm = llm
        self.scraper = HTMLContentScraper() # Create instance of HTMLContentScraper

        self.llm.__init__(system_prompt="""
        You are a Website Analyst AI. Your job is to analyze website code 
        and provide a concise summary of the website's purpose and content. 

        ## Instructions:
        - Focus on understanding what the website is about and what it offers.
        - Identify key elements like headings, text content, and potentially important keywords.
        - Aim for a summary of around 100 words or less.
        - Do not include URLs, links, or HTML tags in your response.
        
        ## Example:
        Website code: (HTML code of a website)
        Summary: This website appears to be an online store selling handcrafted jewelry. 
                  It highlights its unique designs, use of natural materials, and secure online shopping experience. 

        Remember: Provide a clear and informative summary based on the provided code. 
        """)

    def run(self) -> str:
        html_content = self.scraper.scrape_and_clean_html(self.url) # Scrape HTML
        if html_content:
            self.llm.add_message("user", f"Analyze this website HTML:\n```html\n{html_content}\n```")
            response = self.llm.run("Provide a concise summary of the website based on the HTML code.")
            return response
        else:
            return "Unable to fetch and analyze the website content." 

# Example Usage (make sure you have a valid 'Gemini' class and API key setup)
if __name__ == "__main__":
    target_url = "https://icrisstudio1.pythonanywhere.com/" 
    llm = Gemini() # Replace with your actual API key
    analyst = WEBAnalyst(target_url, llm)
    summary = analyst.run()
    print(f"Website Summary:\n{summary}")