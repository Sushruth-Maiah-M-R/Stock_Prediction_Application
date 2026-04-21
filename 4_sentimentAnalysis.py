import pandas as pd
import numpy as np

# Load model once (IMPORTANT: huge speed improvement)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

nlp = pipeline(
    "sentiment-analysis",
    model=finbert,
    tokenizer=tokenizer
)


def FinBERT_sentiment_score(news_list):
    """
    Compute average sentiment score of all headlines
    Range: -1 to +1
    """

    scores = []

    for heading in news_list:

        if pd.isna(heading):
            continue

        heading = str(heading).strip()

        if heading == "" or heading == "0":
            continue

        result = nlp(heading)

        label = result[0]["label"].lower()
        score = result[0]["score"]

        if label == "positive":
            scores.append(score)

        elif label == "negative":
            scores.append(-score)

        else:
            scores.append(0)

    if len(scores) == 0:
        return 0

    return np.mean(scores)


news_df = pd.read_csv("news_data.csv")

BERT_sentiment = []

for i in range(len(news_df)):
    news_list = news_df.iloc[i, 1:].tolist()

    score_BERT = FinBERT_sentiment_score(news_list)

    BERT_sentiment.append(score_BERT)

    print(f"Processed row {i+1}/{len(news_df)}")


news_df["FinBERT score"] = BERT_sentiment

news_df.to_csv("sentiment.csv", index=False)

print("sentiment.csv created successfully")