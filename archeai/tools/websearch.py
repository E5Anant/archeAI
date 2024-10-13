from googlesearch import search

def web_search(query:str):
    try:
        results = list(search(query, advanced=True, num_results=3))
        output = f"The search results for {query} are as follows: \n\n"
        for i, result in enumerate(results):
            output += f"{i+1}. \nTitle: {result.title}\nDescription: {result.description}\nSource: {result.url}\n\n"
        output += "[END]Search Results[END]"
        return output
    except Exception as e:
        return f"An error occurred during the search: {e}"
    
if __name__=="__main__":
    print(web_search("get the weather in kolkata"))