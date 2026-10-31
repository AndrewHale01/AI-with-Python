from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


RANDOM_STATE = 42
SAMPLE_COUNT = 2_000
EPOCHS = 50


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    coordinates = rng.random((SAMPLE_COUNT, 2), dtype=np.float32)
    labels = (coordinates[:, 0] + coordinates[:, 1] >= 1).astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        coordinates,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            layers.Dense(4, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    test_metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    test_probabilities = model.predict(x_test, verbose=0).ravel()
    test_predictions = (test_probabilities >= 0.5).astype(np.int32)

    print("Передбачений клас для тестових точок:")
    for point, predicted_class in zip(x_test, test_predictions):
        print(
            f"(x1={point[0]:.3f}, x2={point[1]:.3f}) "
            f"-> клас {predicted_class}"
        )

    new_points = np.array(
        [
            [0.1, 0.2],
            [0.8, 0.7],
            [0.4, 0.6],
            [0.9, 0.05],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    new_probabilities = model.predict(new_points, verbose=0).ravel()
    new_predictions = (new_probabilities >= 0.5).astype(np.int32)

    print("\nПеревірка на 5 нових точках:")
    for point, predicted_class in zip(new_points, new_predictions):
        print(
            f"(x1={point[0]:.2f}, x2={point[1]:.2f}) "
            f"-> клас {predicted_class}"
        )

    print(f"\nТочність на тестовій вибірці: {test_metrics['accuracy']:.4f}")
    print(f"Функція втрат на тестовій вибірці: {test_metrics['loss']:.4f}")
    print(f"Accuracy score: {metrics.accuracy_score(y_test, test_predictions):.4f}")

    epochs = range(1, EPOCHS + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))

    loss_axis.plot(epochs, history.history["loss"], label="Навчання")
    loss_axis.plot(epochs, history.history["val_loss"], label="Валідація")
    loss_axis.set_title("Функція втрат")
    loss_axis.set_xlabel("Епоха")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    loss_axis.grid(alpha=0.3)

    accuracy_axis.plot(epochs, history.history["accuracy"], label="Навчання")
    accuracy_axis.plot(epochs, history.history["val_accuracy"], label="Валідація")
    accuracy_axis.set_title("Точність")
    accuracy_axis.set_xlabel("Епоха")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
