# Used Car Price Prediction

A machine learning project that predicts used car selling prices based on vehicle characteristics using Python and Scikit-learn.

## Project Overview

This project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Machine learning model comparison
- Model evaluation using Mean Absolute Error (MAE)
- A command-line interface for price prediction

## Dataset

The dataset used in this project is `CAR DETAILS FROM CAR DEKHO.csv`, downloaded from a GitHub repository.

**GitHub dataset source** 

- Original records: 4,340
- Records after removing exact duplicates: 3,577
- Target: `selling_price`
- Currency: Indian Rupees (INR)

The dataset contains information about vehicle names, manufacturing years, mileage, fuel types, transmission types, sellers, ownership history, and selling prices.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Features

The improved model uses eight input features:

1. Manufacturing year
2. Kilometers driven
3. Brand
4. Approximate model name
5. Fuel type
6. Transmission
7. Seller type
8. Ownership history

Categorical data is processed using One-Hot Encoding in a Scikit-learn Pipeline.

## Models Evaluated

| Model | Validation MAE (INR) |
|---|---:|
| Constant-median baseline | 283,304.75 |
| Decision Tree (2 features) | 225,496.47 |
| Random Forest (2 features) | 254,649.14 |
| Random Forest (7 features) | 167,561.88 |
| Random Forest (8 features) | **159,719.52** |
| Extra Trees (8 features) | 166,471.74 |

The best observed validation result was achieved by the eight-feature Random Forest model.

A separate five-fold cross-validation experiment with the seven-feature Random Forest achieved an average MAE of 158,385.39 INR.

These experiments were conducted during model development. An independent untouched test-set evaluation has not yet been completed.

## Installation

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

The dataset must be placed in the `data` directory using the filename `CAR DETAILS FROM CAR DEKHO.csv`.

## Run Price Prediction

```bash
python predict.py
```

Enter the requested vehicle specifications. The application loads the trained model and displays an estimated price in Indian Rupees.

## Limitations

- The dataset contains vehicles manufactured up to 2020.
- Predictions are not current market valuations.
- Brand and model information are extracted using simplified text rules.
- Prediction errors may be substantial for luxury or uncommon vehicles.
- Reported validation metrics may be optimistic due to repeated model selection on the same validation data.

## Future Improvements

- Independent test-set evaluation
- More accurate model and variant extraction
- Additional feature engineering
- Updated vehicle market data
- A web-based prediction interface

## Project Status

Functional machine learning prototype with a trained model and command-line prediction interface.