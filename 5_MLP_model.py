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
learning_rate = 0.01


# -----------------------------
# Load stock price data
# -----------------------------
stock_data = pd.read_csv("stock_price.csv")

print("Columns found:", stock_data.columns.tolist())

stock_data["Close"] = pd.to_numeric(
    stock_data["Close"],
    errors="coerce"
)

stock_data = stock_data.dropna(subset=["Close"]).reset_index(drop=True)

column = ["Close"]

len_stock_data = stock_data.shape[0]


# -----------------------------
# Train / Test split
# -----------------------------
train_examples = int(len_stock_data * split)

train = stock_data[column].values[:train_examples]
test = stock_data[column].values[train_examples:]

len_train = len(train)
len_test = len(test)


# -----------------------------
# Normalize
# -----------------------------
scaler = MinMaxScaler()

train = scaler.fit_transform(train)
test = scaler.transform(test)


# -----------------------------
# Create sequences
# -----------------------------
X_train = []
y_train = []

for i in range(sequence_length, len_train):
    X_train.append(train[i-sequence_length:i, 0])
    y_train.append(train[i, 0])

X_test = []
y_test = []

for i in range(sequence_length, len_test):
    X_test.append(test[i-sequence_length:i, 0])
    y_test.append(test[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# -----------------------------
# Build MLP Model
# -----------------------------
def model_create():

    tf.random.set_seed(1234)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(sequence_length,)),

        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dropout(0.05),

        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.01),

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

results_path = os.path.join(folder, "mlp_results.csv")
metrics_path = os.path.join(folder, "mlp_metrics.csv")

results_df.to_csv(results_path, index=False)
metrics_df.to_csv(metrics_path, index=False)

print("Saved:", results_path)
print("Saved:", metrics_path)