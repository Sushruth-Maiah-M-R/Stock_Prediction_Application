import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load CSV files created earlier
# -----------------------------
mlp_results = pd.read_csv("mlp_results.csv")
lstm_results = pd.read_csv("lstm_results.csv")
bert_results = pd.read_csv("bert_lstm_results.csv")

mlp_metrics = pd.read_csv("mlp_metrics.csv")
lstm_metrics = pd.read_csv("lstm_metrics.csv")
bert_metrics = pd.read_csv("bert_lstm_metrics.csv")


# -----------------------------
# Extract values
# -----------------------------
test = mlp_results["Actual"].values

mlp_preds = mlp_results["Predicted"].values
lstm_preds = lstm_results["Predicted"].values
bert_preds = bert_results["Predicted"].values


# -----------------------------
# Metrics
# -----------------------------
mlp_mae = mlp_metrics["MAE"][0]
mlp_mape = mlp_metrics["MAPE"][0]
mlp_acc = mlp_metrics["Accuracy"][0]

lstm_mae = lstm_metrics["MAE"][0]
lstm_mape = lstm_metrics["MAPE"][0]
lstm_acc = lstm_metrics["Accuracy"][0]

bert_mae = bert_metrics["MAE"][0]
bert_mape = bert_metrics["MAPE"][0]
bert_acc = bert_metrics["Accuracy"][0]


# -----------------------------
# Print Results
# -----------------------------
print("MLP")
print("MAE =", mlp_mae)
print("MAPE =", mlp_mape)
print("Accuracy =", mlp_acc)

print("\nLSTM")
print("MAE =", lstm_mae)
print("MAPE =", lstm_mape)
print("Accuracy =", lstm_acc)

print("\nFinBERT-LSTM")
print("MAE =", bert_mae)
print("MAPE =", bert_mape)
print("Accuracy =", bert_acc)


# -----------------------------
# Main Comparison Plot
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(test, linewidth=2.6, color="black")
plt.plot(mlp_preds, color="green")
plt.plot(lstm_preds, color="orange")
plt.plot(bert_preds, color="red")

plt.xlabel("Timestep", fontsize=10)
plt.ylabel("Closing Price in USD", fontsize=10)
plt.title("NDX Closing Price Prediction Comparison", fontsize=18)

plt.legend([
    "Actual Price",
    "MLP Model",
    "LSTM Model",
    "FinBERT-LSTM Model"
])

plt.grid(alpha=0.3)
plt.show()


# -----------------------------
# MAE Bar Chart
# -----------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    ["MLP", "LSTM", "FinBERT-LSTM"],
    [mlp_mae, lstm_mae, bert_mae]
)

plt.title("MAE Comparison")
plt.ylabel("Error")
plt.show()


# -----------------------------
# Accuracy Bar Chart
# -----------------------------
plt.figure(figsize=(10, 5))

plt.bar(
    ["MLP", "LSTM", "FinBERT-LSTM"],
    [mlp_acc, lstm_acc, bert_acc]
)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()