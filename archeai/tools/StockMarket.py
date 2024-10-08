import yfinance as yf
import datetime

class StockMarketInfo:
    """
    Provides stock market information using the yfinance library. 
    """

    def __init__(self):
        pass  

    def get_stock_price(self, ticker: str) -> float:
        """
        Fetches and returns the current price of a given stock ticker.

        Args:
            ticker (str): The stock symbol (e.g., "AAPL" for Apple).

        Returns:
            float: The current stock price, or None if an error occurs.
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('currentPrice', None)
        except Exception as e:
            print(f"Error fetching stock price: {e}")
            return None

    def get_historical_data(self, ticker: str, period: str = "1y") -> yf.Ticker:
        """
        Retrieves historical stock data for a given period.

        Args:
            ticker (str): The stock symbol.
            period (str, optional): The data period (e.g., "1d", "5d", "1mo", "1y", "5y", "max"). 
                                   Defaults to "1y".

        Returns:
            pandas.DataFrame: A DataFrame containing historical data (Open, High, Low, Close, Volume)
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def get_company_info(self, ticker: str) -> dict:
        """
        Returns basic company information.

        Args:
            ticker (str): The stock symbol.

        Returns:
            dict: A dictionary containing company information (e.g., name, sector, website) 
                  or None if an error occurs.
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            print(f"Error fetching company information: {e}")
            return None

    def get_news(self, ticker: str, count=5) -> list:
        """
        Fetches recent news articles related to the stock ticker.

        Args:
            ticker (str): The stock symbol.
            count (int, optional): The maximum number of news articles to retrieve. 

        Returns:
            list: A list of dictionaries, where each dictionary represents a news article
                  and contains 'title', 'link', and 'published_at' keys.
        """
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:count]

    def get_stock_details(self, ticker: str) -> str:
        """
        Provides detailed information about a specific stock as a formatted string.

        Args:
            ticker (str): The stock symbol.

        Returns:
            str: A formatted string containing stock details, 
                 or an error message if something goes wrong.
        """
        try:
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)

            # Price Information
            price = self.get_stock_price(ticker)
            price_info = f"Current Price: ${price:.2f}" if price else "Price information not available."

            # Company Information
            info = self.get_company_info(ticker)
            if info:
                company_info = f"""
                Company: {info.get('longName', 'N/A')}
                Sector: {info.get('sector', 'N/A')}
                Industry: {info.get('industry', 'N/A')}
                Website: {info.get('website', 'N/A')}
                Business Summary: {info.get('longBusinessSummary', 'N/A')}
                """
            else:
                company_info = "Company information not available."

            # Recent News
            news = self.get_news(ticker)
            if news:
                news_section = "\n--- Recent News ---\n" + "\n".join(
                    [f"- {article['title']} ({article['link']})" for article in news]
                )
            else:
                news_section = "No recent news found."

            # Combine all sections
            stock_details = f"""
            --- Stock Information ---
            {price_info}

            {company_info}

            {news_section}
            ------------------------------
            """
            return stock_details

        except Exception as e:
            return f"Error fetching stock details: {e}"

# Example Usage (Modified)
if __name__ == "__main__":
    market_info = StockMarketInfo()

    while True:
        query = input("Enter a stock ticker (or 'q' to quit): ")
        print(market_info.get_stock_details(query))
        if query.lower() == 'q':
            break