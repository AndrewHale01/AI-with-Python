from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.callbacks import EarlyStopping


OUTPUT_DIR = Path(__file__).resolve().parent


def load_data():
    """Load the California housing dataset as a Pandas DataFrame."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target
    return df


def preprocess_data(df):
    """Split the data and standardize features using only the training set."""
    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_model(input_shape):
    """Build a dense neural network for continuous-value regression."""
    model = Sequential(
        [
            tf.keras.Input(shape=(input_shape,)),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    """Train the model, reserving part of the training data for validation."""
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the held-out test data."""
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return loss, mae


def plot_loss(history):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Model Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.show()


def plot_predictions(y_test, y_pred):
    """Plot actual versus predicted house prices."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, np.asarray(y_pred).ravel(), alpha=0.5)
    plt.xlabel("Actual Prices ($100,000 units)")
    plt.ylabel("Predicted Prices ($100,000 units)")
    plt.title("Actual vs. Predicted House Prices")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
    plt.show()


def main():
    tf.keras.utils.set_random_seed(42)

    df = load_data()
    X_train, X_test, y_train, y_test, _scaler = preprocess_data(df)
    model = build_model(X_train.shape[1])
    history = train_model(model, X_train, y_train)

    loss, mae = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {loss:.4f}")
    print(f"Test MAE: {mae:.4f} ($100,000 units)")

    plot_loss(history)
    y_pred = model.predict(X_test, verbose=0)
    plot_predictions(y_test, y_pred)


if __name__ == "__main__":
    main()