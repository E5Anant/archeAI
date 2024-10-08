import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

class HTMLContentScraper:
    """
    Scrapes a webpage and removes all CSS and JavaScript content.
    """

    def __init__(self, headers=None):
        """
        Initializes the scraper with optional custom headers.

        Args:
            headers (dict, optional): Custom headers for HTTP requests.
        """

        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }

    def scrape_and_clean_html(self,url):
        """
        Fetches the HTML content, removes CSS/JS, and returns the cleaned HTML.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            str: The cleaned HTML content, or None if an error occurs.
        """

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove <script> tags (for JavaScript)
            for script in soup(["script", "style"]): 
                script.decompose()

            # Remove CSS from inline <style> tags
            for style in soup.find_all('style'):
                style.decompose()

            # Remove HTML comments (often contain CSS or JS)
            for element in soup(text=lambda text: isinstance(text, Comment)):
                element.extract()

            return str(soup)  

        except requests.exceptions.RequestException as e:
            print(f"Error during scraping: {e}")
            return None

# Example Usage
if __name__ == "__main__":
    target_url = "https://icrisstudio1.pythonanywhere.com/" 

    scraper = HTMLContentScraper()
    cleaned_html = scraper.scrape_and_clean_html(target_url)

    if cleaned_html:
        print(cleaned_html)
        # You can now save cleaned_html to a file or process it further