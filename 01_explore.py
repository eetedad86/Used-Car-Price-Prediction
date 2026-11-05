from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save charts without requiring a graphical window.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "CAR DETAILS FROM CAR DEKHO.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_model_v2.joblib"
FIGURES_DIR = BASE_DIR / "figures"

NUMERIC_FEATURES = ["year", "km_driven"]
CATEGORICAL_FEATURES = ["brand", "fuel", "transmission", "seller_type", "owner"]
FINAL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ["model_name"]
TARGET = "selling_price"
RANDOM_STATE = 42


def make_pipeline(categorical_features, regressor):
    """Combine the original one-hot encoding scheme with a regression model."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )
    return Pipeline([("preprocessing", preprocessing), ("model", regressor)])


def evaluate(model, features, target, train_idx, valid_idx, label):
    """Fit on training records and report MAE on validation records."""
    model.fit(features.loc[train_idx], target.loc[train_idx])
    predictions = model.predict(features.loc[valid_idx])
    mae = mean_absolute_error(target.loc[valid_idx], predictions)
    print(f"{label}: {mae:,.2f} INR")
    return mae


def save_charts(df):
    """Generate the three plots from the original exploratory analysis."""
    FIGURES_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="year", y=TARGET, hue="transmission", alpha=0.6
    )
    plt.title("Car Year vs Selling Price")
    plt.xlabel("Manufacturing Year")
    plt.ylabel("Selling Price (INR)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "year_vs_price.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="km_driven", y=TARGET, hue="transmission", alpha=0.6
    )
    plt.title("Mileage vs Selling Price")
    plt.xlabel("Kilometers Driven")
    plt.ylabel("Selling Price (INR)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mileage_vs_price.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df[TARGET], bins=30)
    plt.title("Selling Price Distribution")
    plt.xlabel("Selling Price (INR)")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=150)
    plt.close()

    print(f"Charts saved to: {FIGURES_DIR}")


def main():
    if not DATA_PATH.is_file():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the source CSV and put it inside the data/ folder. "
            "Prediction with an existing .joblib model does not need this CSV."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {
        "name", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", TARGET,
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"Missing CSV columns: {sorted(missing_columns)}")

    print("\n--- Original Dataset ---")
    print(f"Shape: {df.shape}")
    print(df.head().to_string(index=False))
    print("\n--- Missing Values ---")
    print(df.isna().sum())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")

    df_clean = df.drop_duplicates().copy()
    if df_clean[list(required_columns)].isna().any().any():
        raise SystemExit("Dataset has missing values in required columns; review them before training.")

    # Keep feature extraction identical to the saved v2 model and predict.py.
    df_clean["brand"] = df_clean["name"].str.split().str[0]
    df_clean["model_name"] = (
        df_clean["name"].str.split().str[:2].str.join(" ")
    )

    print("\n--- Clean Dataset ---")
    print(f"Shape: {df_clean.shape}")
    print("\n--- Category Counts ---")
    for column in ["fuel", "transmission", "seller_type", "owner"]:
        print(f"\n{column}:\n{df_clean[column].value_counts()}")

    print("\n--- Price by Transmission ---")
    print(df_clean.groupby("transmission")[TARGET].agg(["count", "mean", "median"]))
    print("\n--- Price by Year ---")
    print(df_clean.groupby("year")[TARGET].agg(["count", "mean", "median"]))
    print("\n--- Average Mileage by Transmission ---")
    print(df_clean.groupby("transmission")["km_driven"].agg(["mean", "median"]))
    print("\n--- Mileage / Price Correlation ---")
    print(df_clean[["km_driven", TARGET]].corr())
    print("\n--- Five Most Expensive Cars ---")
    print(
        df_clean.nlargest(5, TARGET)[
            ["name", "transmission", "km_driven", TARGET, "year"]
        ].to_string(index=False)
    )
    cheap_pct = 100 * (df_clean[TARGET] < 500_000).mean()
    print(f"\nCars priced below 500,000 INR: {cheap_pct:.2f}%")
    save_charts(df_clean)

    # Use the same records for every train/validation comparison.
    train_idx, valid_idx = train_test_split(
        df_clean.index, test_size=0.2, random_state=RANDOM_STATE
    )
    y = df_clean[TARGET]
    X_two = df_clean[NUMERIC_FEATURES]
    X_seven = df_clean[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_eight = df_clean[NUMERIC_FEATURES + FINAL_CATEGORICAL_FEATURES]

    print("\n--- Model Comparison (same 80/20 split) ---")
    median_price = y.loc[train_idx].median()
    baseline_mae = mean_absolute_error(
        y.loc[valid_idx], [median_price] * len(valid_idx)
    )
    print(f"Median baseline: {baseline_mae:,.2f} INR")

    evaluate(
        DecisionTreeRegressor(max_leaf_nodes=20, random_state=RANDOM_STATE),
        X_two, y, train_idx, valid_idx, "Decision Tree (2 features)",
    )
    evaluate(
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        X_two, y, train_idx, valid_idx, "Random Forest (2 features)",
    )

    rf_seven = make_pipeline(
        CATEGORICAL_FEATURES,
        RandomForestRegressor(
            n_estimators=100, max_leaf_nodes=40, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )
    evaluate(rf_seven, X_seven, y, train_idx, valid_idx, "Random Forest (7 features)")

    rf_eight = make_pipeline(
        FINAL_CATEGORICAL_FEATURES,
        RandomForestRegressor(
            n_estimators=100, max_leaf_nodes=40, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )
    evaluate(rf_eight, X_eight, y, train_idx, valid_idx, "Random Forest (8 features)")

    extra_trees = make_pipeline(
        FINAL_CATEGORICAL_FEATURES,
        ExtraTreesRegressor(
            n_estimators=100, max_leaf_nodes=40,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )
    evaluate(extra_trees, X_eight, y, train_idx, valid_idx, "Extra Trees (8 features)")

    # This is cross-validation for v2, not for the older 7-feature model.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_maes = -cross_val_score(
        rf_eight, X_eight, y, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    print("\n--- Final Model: 5-Fold Cross-Validation ---")
    print("Fold MAEs:", cv_maes)
    print(f"Average CV MAE: {cv_maes.mean():,.2f} INR")
    print(
        "Note: These are development estimates, not an independent untouched "
        "test-set result."
    )

    # Retrain the chosen configuration on all cleaned data for predictions.
    final_model = clone(rf_eight)
    final_model.fit(X_eight, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    dump(final_model, MODEL_PATH)
    print(f"\nImproved model saved successfully: {MODEL_PATH}")

