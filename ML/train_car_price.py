from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


NUMERIC_FEATURES = ["year", "engine_volume", "mileage", "horsepower"]
CATEGORICAL_FEATURES = ["brand", "model"]
TARGET = "price"
DATA_PATH = Path(__file__).resolve().parent.parent / "assets" / "cars.csv"
DEMO_DATA_PATH = Path(__file__).resolve().parent / "cars_with_brand_model.csv"
PLOT_PATH = Path(__file__).resolve().parent / "actual_vs_predicted.png"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required_columns = NUMERIC_FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"У CSV відсутні стовпці: {', '.join(missing_columns)}")

    data[required_columns] = data[required_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    data = data.dropna(subset=required_columns)
    if data.empty:
        raise ValueError("У CSV немає повних числових рядків для навчання.")

    present_categories = set(CATEGORICAL_FEATURES) & set(data.columns)
    if present_categories and len(present_categories) != len(CATEGORICAL_FEATURES):
        missing_categories = sorted(set(CATEGORICAL_FEATURES) - present_categories)
        raise ValueError(
            f"Додайте до CSV обидва категоріальні стовпці: "
            f"{', '.join(missing_categories)}"
        )

    if not present_categories:
        brands_and_models = {
            "Toyota": (("Corolla", 0), ("Camry", 12_000)),
            "BMW": (("3 Series", 18_000), ("5 Series", 32_000)),
            "Volkswagen": (("Golf", 4_000), ("Passat", 14_000)),
            "Hyundai": (("Elantra", -8_000), ("Tucson", 6_000)),
        }
        brand_premiums = {
            "Toyota": 4_000,
            "BMW": 35_000,
            "Volkswagen": 10_000,
            "Hyundai": -5_000,
        }
        rng = np.random.default_rng(42)
        brands = list(brands_and_models)
        data["brand"] = rng.choice(brands, size=len(data))
        selected_models = [
            brands_and_models[brand][rng.integers(0, 2)]
            for brand in data["brand"]
        ]
        data["model"] = [model for model, _ in selected_models]
        model_premiums = {
            (brand, model): premium
            for brand, choices in brands_and_models.items()
            for model, premium in choices
        }
        data[TARGET] += data["brand"].map(brand_premiums)
        data[TARGET] += [
            model_premiums[(brand, model)]
            for brand, model in zip(data["brand"], data["model"])
        ]
        data.to_csv(DEMO_DATA_PATH, index=False)
        print(
            "CSV has no brand/model columns; generated demo categories and "
            f"simulated price effects in {DEMO_DATA_PATH}"
        )

    data = data.dropna(subset=CATEGORICAL_FEATURES)
    if data.empty:
        raise ValueError("У CSV немає рядків із повними категоріальними ознаками.")
    return data


def main() -> None:
    data = load_data()
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    prices = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, prices, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    predicted_prices = model.predict(x_test)

    mae = mean_absolute_error(y_test, predicted_prices)
    mape = mean_absolute_percentage_error(y_test, predicted_prices) * 100
    r2 = r2_score(y_test, predicted_prices)

    print(f"Dataset rows: {len(data)}")
    print(f"Mean absolute error (MAE): {mae:,.2f} UAH")
    print(f"Mean absolute percentage error (MAPE): {mape:.2f}%")
    print(f"Coefficient of determination (R2): {r2:.3f}")

    lower_bound = min(y_test.min(), predicted_prices.min())
    upper_bound = max(y_test.max(), predicted_prices.max())
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predicted_prices, alpha=0.75, edgecolors="black")
    plt.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        "r--",
        label="Ідеальний прогноз",
    )
    plt.xlabel("Справжня ціна (грн)")
    plt.ylabel("Прогнозована ціна (грн)")
    plt.title("Справжня vs Прогнозована ціна")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Plot saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
