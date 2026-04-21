# actual_vs_predicted_scatter.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load Files
# -----------------------------
mlp_results = pd.read_csv("mlp_results.csv")
lstm_results = pd.read_csv("lstm_results.csv")
bert_results = pd.read_csv("bert_lstm_results.csv")

# -----------------------------
# Extract Data
# -----------------------------
actual = mlp_results["Actual"].values

mlp_pred = mlp_results["Predicted"].values
lstm_pred = lstm_results["Predicted"].values
bert_pred = bert_results["Predicted"].values

# -----------------------------
# Perfect Prediction Line
# -----------------------------
min_val = min(actual.min(), mlp_pred.min(), lstm_pred.min(), bert_pred.min())
max_val = max(actual.max(), mlp_pred.max(), lstm_pred.max(), bert_pred.max())

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 8))

plt.scatter(actual, mlp_pred, alpha=0.7, label="MLP")
plt.scatter(actual, lstm_pred, alpha=0.7, label="LSTM")
plt.scatter(actual, bert_pred, alpha=0.7, label="FinBERT-LSTM")

# Perfect diagonal line
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

# -----------------------------
# Labels
# -----------------------------
plt.title("Actual vs Predicted Price Scatter Plot")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(alpha=0.3)

plt.show()