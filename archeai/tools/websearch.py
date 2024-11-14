import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def web_search(query, num_results=5, char_limit=1000):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        results = []
        for g in soup.find_all("div", class_="tF2Cxc"):
            a_tag = g.find("a")
            if a_tag:
                link = a_tag["href"]
                title = g.find("h3").text if g.find("h3") else ""

                try:
                    content = extract_content(link, char_limit)
                    if content:  # Check if content extraction was successful
                        results.append({"title": title, "url": link, "content": content})
                        if len(results) >= num_results:
                            break
                except Exception as e:
                    print(f"Error fetching or processing content from {link}: {e}")

        
        for result in results:
            return f"Title: {result['title']}\n \nURL: {result['url']}\n \nContent: {result['content']}\n\n"

    except requests.exceptions.RequestException as e:
        print(f"Error during search: {e}")
        return []


def extract_content(url, char_limit):

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        text = "".join(soup.stripped_strings)
        return text[:char_limit]

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""  # Return empty on error


if __name__=="__main__":
    # Example usage:
    search_results = web_search("Python web scraping tutorials", num_results=2, char_limit=2000)

    print(search_results)
