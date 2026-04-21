import pandas as pd

def download_stock_data(ticker, start, end):
    """
    Download clean stock price data from Yahoo Finance
    """

    import yfinance as yf

    stock_data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False
    )

    stock_data.reset_index(inplace=True)

    # flatten columns if multi-index appears
    stock_data.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in stock_data.columns
    ]

    stock_data.to_csv("stock_price.csv", index=False)

    print("stock_price.csv created successfully")
    print(stock_data.head())


download_stock_data("QQQ", "2025-04-25", "2026-04-20")