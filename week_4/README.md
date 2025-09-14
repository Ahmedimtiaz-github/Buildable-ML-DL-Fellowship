# Week 4 – Buildable ML/DL Fellowship

This week contains two machine learning tasks:

## Classification (Weather Data)
- Dataset: Weather conditions
- Target: `Weather Type` (categorical)
- Key Steps:
  - Cleaning anomalies (humidity > 100, wind speed outliers, etc.)
  - Feature engineering
  - Train/test split with stratification
  - Models: Logistic Regression, Decision Tree, Random Forest
  - Evaluation: Accuracy, confusion matrices, classification reports
- Outputs:
  - Cleaned dataset (`classification_cleaned.csv`)
  - Figures in `classification/figures/`

## Regression (PakWheels Used Cars)
- Dataset: PakWheels used cars
- Target: `Price` (continuous)
- Key Steps:
  - Cleaning numeric + categorical features (price, mileage, make, model)
  - Feature engineering (log(price), capping outliers)
  - Train/test split
  - Model: Random Forest Regressor
  - Evaluation: RMSE, residuals plot, actual vs predicted
- Outputs:
  - Cleaned dataset (`regression_cleaned.csv`)
  - Figures in `regression/figures/`

---

✅ Both tasks include cleaned data, plots, and Jupyter notebooks.  
