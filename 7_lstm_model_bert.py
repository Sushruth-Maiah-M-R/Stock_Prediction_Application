import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
import os

# hyperparameters
split = 0.85
sequence_length = 10
epochs = 100
learning_rate = 0.02


# -----------------------------
# Load stock + sentiment data
# -----------------------------
stock_data = pd.read_csv("stock_price.csv")
news_data = pd.read_csv("sentiment.csv")

stock_data["Close"] = pd.to_numeric(
    stock_data["Close"],
    errors="coerce"
)

news_data["FinBERT score"] = pd.to_numeric(
    news_data["FinBERT score"],
    errors="coerce"
)

stock_data = stock_data.dropna(subset=["Close"]).reset_index(drop=True)
news_data = news_data.dropna(subset=["FinBERT score"]).reset_index(drop=True)

# align rows
min_len = min(len(stock_data), len(news_data))

stock_data = stock_data.iloc[:min_len]
news_data = news_data.iloc[:min_len]


# -----------------------------
# Extract values
# -----------------------------
prices = stock_data["Close"].values.reshape(-1, 1)
sentiments = news_data["FinBERT score"].values.reshape(-1, 1)


# -----------------------------
# Train/Test Split
# -----------------------------
train_examples = int(min_len * split)

train = prices[:train_examples]
test = prices[train_examples:]

train_sentiment = sentiments[:train_examples]
test_sentiment = sentiments[train_examples:]


# -----------------------------
# Normalize price only
# -----------------------------
scaler = MinMaxScaler()

train = scaler.fit_transform(train)
test = scaler.transform(test)


# -----------------------------
# Create train sequences
# -----------------------------
X_train = []
y_train = []

for i in range(sequence_length, len(train)):

    seq = train[i-sequence_length:i, 0].tolist()
    seq.append(train_sentiment[i][0])

    X_train.append(seq)
    y_train.append(train[i][0])


# -----------------------------
# Create test sequences
# -----------------------------
X_test = []
y_test = []

for i in range(sequence_length, len(test)):

    seq = test[i-sequence_length:i, 0].tolist()
    seq.append(test_sentiment[i][0])

    X_test.append(seq)
    y_test.append(test[i][0])


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# -----------------------------
# Build Model
# -----------------------------
def model_create():

    tf.random.set_seed(1234)

    model = tf.keras.models.Sequential([

        tf.keras.Input(shape=(X_train.shape[1], 1)),

        tf.keras.layers.LSTM(
            70,
            activation="tanh",
            return_sequences=True
        ),

        tf.keras.layers.LSTM(
            30,
            activation="tanh",
            return_sequences=True
        ),

        tf.keras.layers.LSTM(
            10,
            activation="tanh"
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=1
    )

    return model


# -----------------------------
# Real values
# -----------------------------
y_test_real = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# -----------------------------
# Predict
# -----------------------------
def predict(model):

    pred = model.predict(X_test)

    pred = scaler.inverse_transform(pred)

    return pred


# -----------------------------
# Evaluate
# -----------------------------
def evaluate(pred):

    mae = mean_absolute_error(y_test_real, pred)
    mape = mean_absolute_percentage_error(y_test_real, pred)
    acc = 1 - mape

    return mae, mape, acc


# -----------------------------
# Run Model
# -----------------------------
def run_model(n):

    total_mae = 0
    total_mape = 0
    total_acc = 0

    final_pred = None

    for i in range(n):

        print(f"Run {i+1}/{n}")

        model = model_create()

        pred = predict(model)

        mae, mape, acc = evaluate(pred)

        total_mae += mae
        total_mape += mape
        total_acc += acc

        final_pred = pred

    return (
        total_mae / n,
        total_mape / n,
        total_acc / n,
        final_pred
    )


# -----------------------------
# Execute
# -----------------------------
mae, mape, acc, preds = run_model(1)

print("\nResults")
print("Mean Absolute Error =", mae)
print("Mean Absolute Percentage Error =", mape)
print("Accuracy =", acc)


# -----------------------------
# Save CSV Files
# -----------------------------
folder = os.getcwd()

print("Saving to:", folder)

results_df = pd.DataFrame({
    "Actual": y_test_real.flatten(),
    "Predicted": preds.flatten()
})

metrics_df = pd.DataFrame({
    "MAE": [mae],
    "MAPE": [mape],
    "Accuracy": [acc]
})

results_path = os.path.join(folder, "bert_lstm_results.csv")
metrics_path = os.path.join(folder, "bert_lstm_metrics.csv")

results_df.to_csv(results_path, index=False)
metrics_df.to_csv(metrics_path, index=False)

print("Saved:", results_path)
print("Saved:", metrics_path)