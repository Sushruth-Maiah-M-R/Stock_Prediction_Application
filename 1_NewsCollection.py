# Finnhub Historical Finance News
# Correct historical version using company-news

import requests
import pandas as pd
import numpy as np
import time
from datetime import timedelta

API_KEY = "d7jknuhr01qhf13f8dbgd7jknuhr01qhf13f8dc0"

# Better symbols for Nasdaq sentiment
SYMBOLS = ["QQQ", "AAPL", "MSFT", "NVDA"]


def get_news(date_obj):
    """
    Fetch previous 7 days historical finance news
    """
    news_list = []
    seen = set()

    to_date = date_obj.strftime("%Y-%m-%d")
    from_date = (date_obj - timedelta(days=7)).strftime("%Y-%m-%d")

    while True:
        try:
            for symbol in SYMBOLS:

                url = "https://finnhub.io/api/v1/company-news"

                params = {
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": API_KEY
                }

                r = requests.get(url, params=params, timeout=20)

                if r.status_code == 429:
                    print(f"Rate limit hit on {to_date}. Cooling down 60 sec...")
                    time.sleep(60)
                    raise Exception("retry")

                if r.status_code != 200:
                    continue

                data = r.json()

                if not isinstance(data, list):
                    continue

                for article in data:
                    title = article.get("headline", "").replace(",", "").strip()

                    if title and title not in seen:
                        news_list.append(title)
                        seen.add(title)

                    if len(news_list) == 10:
                        return news_list

                time.sleep(1)

            return news_list

        except Exception:
            time.sleep(30)


def generate_news_file():
    start = "2025-04-25"
    end = "2026-04-20"

    dates = pd.date_range(start, end)

    matrix = np.empty((len(dates) + 1, 11), dtype=object)
    matrix[:] = ""

    matrix[0][0] = "Date"

    for i in range(10):
        matrix[0][i + 1] = f"News {i+1}"

    for idx, dt in enumerate(dates):
        date_txt = dt.strftime("%Y-%m-%d")

        print("Processing:", date_txt)

        matrix[idx + 1][0] = date_txt

        news = get_news(dt)

        for j in range(len(news)):
            matrix[idx + 1][j + 1] = news[j]

        if (idx + 1) % 5 == 0:
            pd.DataFrame(matrix).to_csv("news.csv", index=False, header=False)
            print("Progress saved.")

        time.sleep(2)

    df = pd.DataFrame(matrix)
    df.to_csv("news.csv", index=False, header=False)

    print("news.csv created successfully")


generate_news_file()