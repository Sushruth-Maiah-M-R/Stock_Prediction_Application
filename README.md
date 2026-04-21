# Stock Market Prediction using FinBERT + LSTM

## Overview

This project predicts stock/index closing prices using Machine Learning and Deep Learning models. It combines historical market price data with financial news sentiment analysis to improve forecasting accuracy.

The system compares three models:

- MLP (Multi Layer Perceptron)
- LSTM (Long Short-Term Memory)
- FinBERT + LSTM Hybrid Model

The hybrid model uses financial news sentiment extracted using FinBERT along with historical prices.

---

## Problem Statement

Traditional stock prediction models rely only on historical prices. Market movement is also influenced by news, sentiment, and investor psychology.

This project aims to improve prediction performance by combining:

- Technical price history
- Financial news sentiment
- Deep learning sequence modeling

---

## Features

- Historical stock/index price prediction
- Financial news sentiment analysis using FinBERT
- Comparison of multiple ML/DL models
- Candlestick chart with prediction overlay
- Actual vs Predicted visualizations
- Error metric comparison
- CSV export of predictions and metrics
- Modular Python scripts

---

## Models Used

### 1. MLP Model

Uses previous closing prices as input and predicts future price using fully connected neural networks.

### 2. LSTM Model

Uses sequential historical data and learns temporal dependencies in stock prices.

### 3. FinBERT + LSTM Model

Uses:

- Historical closing prices
- Financial news sentiment scores

This hybrid model achieved the best performance among tested models.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Plotly
- HuggingFace Transformers
- FinBERT
- yFinance

---

## Project Structure

```text
Stock_Price_Prediction/
│── venv/
│
│── 1_NewsCollection.py
│── 2_stockMarket.py
│── 3_newsDataCleaning.py
│── 4_sentimentAnalysis.py
│── 5_MLP_model.py
│── 6_LSTM_model.py
│── 7_lstm_model_bert.py
│
│── analysis.py
│── actual_vs_predicted_scatter.py
│
│── mlp_results.csv
│── mlp_metrics.csv
│── lstm_results.csv
│── lstm_metrics.csv
│── bert_lstm_results.csv
│── bert_lstm_metrics.csv
│
│── news.csv
│── news_data.csv
│── sentiment.csv
│── stock_price.csv
│
│── README.md
