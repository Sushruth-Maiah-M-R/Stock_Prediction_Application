# candlestick_prediction.py

import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Load files
# -----------------------------
stock_data = pd.read_csv("stock_price.csv")
bert_results = pd.read_csv("bert_lstm_results.csv")

# -----------------------------
# Clean stock data
# -----------------------------
stock_data["Date"] = pd.to_datetime(stock_data["Date"])

stock_data["Open"] = pd.to_numeric(stock_data["Open"], errors="coerce")
stock_data["High"] = pd.to_numeric(stock_data["High"], errors="coerce")
stock_data["Low"] = pd.to_numeric(stock_data["Low"], errors="coerce")
stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors="coerce")

stock_data = stock_data.dropna().reset_index(drop=True)

# -----------------------------
# Match test section length
# -----------------------------
pred_len = len(bert_results)

plot_data = stock_data.tail(pred_len).copy()

plot_data["Predicted"] = bert_results["Predicted"].values

# -----------------------------
# Create chart
# -----------------------------
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=plot_data["Date"],
    open=plot_data["Open"],
    high=plot_data["High"],
    low=plot_data["Low"],
    close=plot_data["Close"],
    name="Actual Price"
))

# Prediction line
fig.add_trace(go.Scatter(
    x=plot_data["Date"],
    y=plot_data["Predicted"],
    mode="lines",
    name="FinBERT-LSTM Prediction",
    line=dict(width=3)
))

# -----------------------------
# Layout
# -----------------------------
fig.update_layout(
    title="Candlestick Chart with Prediction Overlay",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=700
)

fig.show()