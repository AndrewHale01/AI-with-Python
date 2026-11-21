from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


OUTPUT_DIR = Path(__file__).resolve().parent
EPOCHS = 5
BATCH_SIZE = 128
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_data():
    """Load Fashion-MNIST and normalize image pixels to the [0, 1] range."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
    x_test = x_test[..., tf.newaxis].astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Build a convolutional neural network for 10 clothing categories."""
    model = models.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Plot and save training/validation accuracy and loss."""
    epochs = range(len(history.history["accuracy"]))
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4))

    accuracy_axis.plot(epochs, history.history["accuracy"], label="Train accuracy")
    accuracy_axis.plot(
        epochs, history.history["val_accuracy"], label="Validation accuracy"
    )
    accuracy_axis.set_title("Accuracy over epochs")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)

    loss_axis.plot(epochs, history.history["loss"], label="Train loss")
    loss_axis.plot(epochs, history.history["val_loss"], label="Validation loss")
    loss_axis.set_title("Loss over epochs")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    figure.tight_layout()
    chart_path = OUTPUT_DIR / "fashion_mnist_training.png"
    figure.savefig(chart_path, dpi=150)
    print(f"Training chart saved to: {chart_path}")
    plt.show()


def main():
    tf.keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    plot_history(history)


if __name__ == "__main__":
    main()