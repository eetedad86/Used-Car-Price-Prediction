import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from joblib import dump
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor




BASE_DIR = Path(__file__).resolve().parent

csv_path = BASE_DIR / "data" / "CAR DETAILS FROM CAR DEKHO.csv"

df = pd.read_csv(csv_path)

print(df.head())
print(df.shape)

print("\n--- Column Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Duplicate Records ---")
duplicated_rows = df[df.duplicated(keep=0)]

print(duplicated_rows.head(15).to_string(index=0))

print("\n--- Unique Records ---")
print(len(df.drop_duplicates()))

# Create a cleaned copy of the dataset
df_clean = df.drop_duplicates()

print("\n--- Clean Dataset ---")
print(df_clean.shape)

print("\n--- Fuel Types ---")
print(df_clean.fuel.value_counts())

print("\n--- Transmission Types ---")
print(df_clean.transmission.value_counts())

print("\n--- Seller Types ---")
print(df_clean.seller_type.value_counts())

print("\n--- Owner Types ---")
print(df_clean.owner.value_counts())

print("\n--- Price by Transmission Type ---")
print(
    df_clean.groupby('transmission').selling_price
    .agg(['count', 'mean', 'median'])
    )

print("\n--- Price by Year Type ---")
print(
    df_clean.groupby('year').selling_price
    .agg(['count', 'mean', 'median'])
)

print("\n--- Year by Transmission Type ---")
print(
    df_clean.groupby('transmission').year
    .agg(['count', 'mean', 'median'])
)

print("\n--- Price by Year and Transmission ---")
result = (
    df_clean.groupby(['year', 'transmission']).selling_price
    .agg(['count', 'mean', 'median'])
)
print(result.loc[[2016, 2017, 2018]])

# Relationship between car year and selling price
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data = df_clean,
    x = "year",
    y = "selling_price",
    hue = "transmission",
    alpha = 0.6
)

plt.title("Car Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price (INR)")

plt.tight_layout()
plt.show()

expensive_cars = df_clean.nlargest(5, "selling_price")
print(
    expensive_cars[["name", "transmission", "km_driven", "selling_price", "year"]]
)

# Chart 2: Mileage vs Price
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data = df_clean,
    x = "km_driven",
    y = "selling_price",
    hue = "transmission",
    alpha = 0.6
)

plt.title("km_driven vs Price")
plt.xlabel("km_driven")
plt.ylabel("price")
plt.tight_layout()
plt.show()

print(
    df_clean.groupby("transmission").km_driven
    .agg(["mean", "median"])
)

print(df_clean[["km_driven", "selling_price"]].corr())

manual_cars = df_clean[df_clean.transmission == "Manual"]
print(
    manual_cars.km_driven.corr(
        manual_cars.selling_price
    )
)

Automatic_cars = df_clean[df_clean.transmission == "Automatic"]
print(
    Automatic_cars.km_driven.corr(
        Automatic_cars.selling_price
    )
)

# Chart 3: Selling Price Distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data = df_clean,
    x = "selling_price",
    bins = 30
)

plt.title("Selling Price Distribution")
plt.xlabel("Selling Price (INR)")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.show()

cheap_cars = df_clean[df_clean.selling_price < 500000]
percentage = (len(cheap_cars) / len(df_clean)) * 100
print(percentage)

#                                     ***************************************************                         
X = df_clean[["km_driven", "year"]]
print(X.head())

y = df_clean.selling_price
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_valid.shape)

model = DecisionTreeRegressor(
    max_leaf_nodes = 20,
    random_state = 42
)
model.fit(X_train, y_train)

perdiction = model.predict(X_valid)
print(perdiction[:5])
print(y_valid.head())

mae = mean_absolute_error(y_valid, perdiction)
print(f"Validation MAE: {mae:.2f} INR")

baseline_price = y_train.median()
baseline_perdiction = [baseline_price] * len(y_valid)
baseline_mae = mean_absolute_error(
    y_valid,
    baseline_perdiction
)
print(f"Baseline MAE: {baseline_mae:.2f} INR")
print(f"Decision Tree MAE: {mae:.2f} INR")

train_perdiction = model.predict(X_train)
train_mae = mean_absolute_error(
    y_train,
    train_perdiction
)
print(f"Training MAE: {train_mae:.2f} INR")
print(f"Validation MAE: {mae:.2f} INR")


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

rf_model.fit(X_train, y_train)
rf_perdiction = rf_model.predict(X_valid)

rf_mae = mean_absolute_error(y_valid, rf_perdiction)

print(f"Decision Tree MAE: {mae:.2f} INR")
print(f"Random Forest MAE: {rf_mae:.2f} INR")



# Extract brand
df_clean["brand"] = df_clean["name"].str.split().str[0]
numeric_features = ["year", "km_driven"]

categorical_features = [
    "brand", "fuel", "transmission",
    "seller_type", "owner"
]

X_full = df_clean[numeric_features + categorical_features]
y_full = df_clean.selling_price

X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(
    X_full, y_full,
    test_size=0.2,
    random_state=42
)

preprocesser = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder = "passthrough"   
)

rf_full = Pipeline([
    ("processing", preprocesser),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_leaf_nodes=40,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ))
])

rf_full.fit(X_train_full, y_train_full)
full_perdiction = rf_full.predict(X_valid_full)

full_mae = mean_absolute_error(y_valid_full, full_perdiction)

print(f"Old Random Forest MAE: {rf_mae:.2f} INR")
print(f"Enhanced Random Forest MAE: {full_mae:.2f} INR")

full_train_predictions = rf_full.predict(X_train_full)
full_train_mae = mean_absolute_error(
    y_train_full,
    full_train_predictions
)

print(f"Training MAE: {full_train_mae:.2f} INR")
print(f"Validation MAE: {full_mae:.2f} INR")


for min_leaf in [1, 3, 5]:

    test_model = clone(rf_full)

    test_model.set_params(
        model__min_samples_leaf=min_leaf
    )

    test_model.fit(X_train_full, y_train_full)

    predictions = test_model.predict(X_valid_full)

    test_mae = mean_absolute_error(
        y_valid_full, predictions
    )

    print(f"min_samples_leaf={min_leaf} | MAE={test_mae:.2f}")


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf_full,
    X_full,
    y_full,
    cv=cv,
    scoring="neg_mean_absolute_error"
)

mae_scores = -scores

print("Cross-Validation MAE:", mae_scores)
print(f"Average CV MAE: {mae_scores.mean():.2f} INR")




from sklearn.base import clone
from joblib import dump
from pathlib import Path

# Train the final model on all cleaned data
final_model = clone(rf_full)
final_model.fit(X_full, y_full)

# Create a folder for the saved model
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(exist_ok=True)

# Save the model
model_path = models_dir / "car_price_model.joblib"
dump(final_model, model_path)

print("Model saved successfully!")
print(model_path)


df_clean["model_name"] = (
    df_clean["name"].str.split().str[:2].str.join(" ")
)

print(df_clean[["name", "model_name"]].head())

new_categorical = categorical_features + ["model_name"]

X_more = df_clean[numeric_features + new_categorical]

X_train_more, X_valid_more, y_train_more, y_valid_more = train_test_split(
    X_more, y_full, test_size=0.2, random_state=42
)

new_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), new_categorical)
    ],
    remainder="passthrough"
)

rf_more = Pipeline([
    ("preprocessing", new_preprocessor),
    ("model", clone(rf_full.named_steps["model"]))
])

rf_more.fit(X_train_more, y_train_more)

new_predictions = rf_more.predict(X_valid_more)

new_mae = mean_absolute_error(y_valid_more, new_predictions)

print(f"Previous MAE: {full_mae:.2f} INR")
print(f"New MAE: {new_mae:.2f} INR")


extra_model = Pipeline([
    ("preprocessing", new_preprocessor),
    ("model", ExtraTreesRegressor(
        n_estimators=100,
        max_leaf_nodes=40,
        random_state=42,
        n_jobs=-1
    ))
])

extra_model.fit(X_train_more, y_train_more)

extra_predictions = extra_model.predict(X_valid_more)

extra_mae = mean_absolute_error(
    y_valid_more,
    extra_predictions
)

print(f"Random Forest MAE: {new_mae:.2f} INR")
print(f"Extra Trees MAE: {extra_mae:.2f} INR")


# Train the improved model
best_model = clone(rf_more)
best_model.fit(X_more, y_full)

# Save the improved model
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "car_price_model_v2.joblib"

dump(best_model, model_path)

print("Improved model saved successfully!")
print(model_path)
