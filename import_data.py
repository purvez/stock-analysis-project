import yfinance as yf
import pandas as pd
from typing import Any, Dict

class StockDataLoader:
    """
    Downloads up to 3 years of daily data for a single ticker,
    keeps the standard columns (Open, High, Low, Close, Volume), and
    returns a cleaned pandas DataFrame.
    """
    def __init__(self, tickers: str = "NVDA"): 
        """
        Initialise loader

        Args:
        tickers(str): Ticker symbol (e.g., "AAPL"). Single symbol expected.

        """
        self.PERIOD = "3y" #Stock prices for 3 years
        self.INTERVAL = "1d" #Daily interval of stock prices
        self.TICKERS = tickers
        
    def load(self):
        """
        Fetch, clean and cache the DataFrame. 

        Returns:
        pd.DataFrame: Cleaned DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
        """
        df = self.fetch_stock_data()  
        df = self.clean_data(df)
        self.df = df
        # df.to_csv("stock_data.csv", index=False)
        return df

    def fetch_stock_data(self):
        """
        Download raw OHLCV data from yfinance, resets the index so Date' becomes a column formatted as MM/DD/YYYY.

        Returns:
        pd.DataFrame: Raw DataFrame with Date and OHLCV columns
        """
        data = yf.download(tickers = self.TICKERS, #Download data
                           period=self.PERIOD,
                           interval=self.INTERVAL,
                           actions=True,  #Use adjusted close
                           multi_level_index = False) #Remove multi index df
        data = data[["Open", "High", "Low", "Close", "Volume"]] #Column headers: Open, High, Low, CLose Volume
        data = data.reset_index()

        return data

    def clean_data(self, df: pd.DataFrame):
        """
        Cleans data by:
        1)deleting rows with missing value
        2)sort dateindex to ascending order
        
        Parameters
        df (pd.DataFrame): Raw stocks data frame

        Returns
        pd.DataFrame: Cleaned stocks data frame
        """
        df = df.dropna() #Delete rows with any missing value
        df = df.sort_index() #Sort the dateindex in ascending order
        return df


# if __name__ == "__main__":
#     loader = StockDataLoader()
#     df = loader.load()
#     loader.description()






