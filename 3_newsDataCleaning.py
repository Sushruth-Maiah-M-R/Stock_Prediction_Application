import pandas as pd

news_df = pd.read_csv("news.csv")

stock_df = pd.read_csv("stock_price.csv", skiprows=2)

# Rename first column to Date
stock_df.rename(columns={stock_df.columns[0]: "Date"}, inplace=True)

# Keep only yyyy-mm-dd
stock_df["Date"] = stock_df["Date"].astype(str).str[:10]

# Filter news rows matching stock dates
news_df = news_df[news_df["Date"].isin(stock_df["Date"].tolist())]

news_df.to_csv("news_data.csv", index=False)

print("news_data.csv created successfully")