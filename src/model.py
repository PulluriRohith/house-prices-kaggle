"""
model.py  – Train XGBoost on the House-Prices data
-------------------------------------------------------
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from xgboost import XGBRegressor
import joblib

from preprocess_house_prices import preprocess_house_prices

# 1) Define your 9 most important features
IMPORTANT = [
    "LotArea", "OverallQual", "BsmtFinSF1", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageCars", "GarageArea"
]

# ------------------------------------------------------------
# 0.  Load raw CSVs
# ------------------------------------------------------------
df_train_raw = pd.read_csv("train.csv")
df_test_raw  = pd.read_csv("test.csv")

# ------------------------------------------------------------
# 1.  Pre-process  (fit on train, reuse on test)
# ------------------------------------------------------------
df_train_proc, stats = preprocess_house_prices(df_train_raw)          # fit+transform
df_test_proc,  _     = preprocess_house_prices(df_test_raw, stats)    # transform-only

# One-hot any leftover object columns, keep train/test aligned
X      = pd.get_dummies(df_train_proc, dtype=float)
X_test = pd.get_dummies(df_test_proc,  dtype=float)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Target (log-scale so we optimise RMSLE)
y = np.log1p(df_train_raw["SalePrice"])

# ------------------------------------------------------------
# 2.  Train / validation split
# ------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------------------------------------
# 3.  XGBoost with Optuna-found hyper-params
# ------------------------------------------------------------
xgb = XGBRegressor(
    random_state    = 42,
    n_estimators    = 235,
    max_depth       = 3,
    learning_rate   = 0.06413266542210438,
    subsample       = 0.9418825300338858,
    colsample_bytree= 0.849657189537435,
    objective       = "reg:squarederror",
    tree_method     = "hist",
    n_jobs          = -1
)

xgb.fit(X_tr, y_tr)

# ------------------------------------------------------------
# 4.  Hold-out evaluation
# ------------------------------------------------------------
val_pred_log = xgb.predict(X_val)

mse   = mean_squared_error(y_val, val_pred_log)
r2_log = r2_score(y_val, val_pred_log)

y_val_raw    = np.expm1(y_val)
val_pred_raw = np.expm1(val_pred_log)

r2_raw  = r2_score(y_val_raw, val_pred_raw)
rmsle   = np.sqrt(mean_squared_log_error(y_val_raw, val_pred_raw))

print("\nValidation metrics")
print("------------------")
print(f"MSE  (log space) : {mse:.4f}")
print(f"R²   (log space) : {r2_log:.4f}")
print(f"R²   (raw scale) : {r2_raw:.4f}")
print(f"RMSLE (raw)      : {rmsle:.5f}\n")

# ------------------------------------------------------------
# 5.  Retrain on all data & create submission
# ------------------------------------------------------------
xgb.fit(X, y)

test_pred = np.expm1(xgb.predict(X_test))
submission = pd.DataFrame({
    "Id": df_test_raw["Id"],
    "SalePrice": test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv written") # This is the file you submit to Kaggle

# ------------------------------------------------------------
# 6.  Persist model (+ preprocessing stats + medians) for later use
# ------------------------------------------------------------
# compute medians for the 9 important features
medians = {f: df_train_raw[f].median() for f in IMPORTANT}

joblib.dump({
    "model":   xgb,
    "stats":   stats,
    "columns": X.columns,
    "medians": medians
}, "xgb_model.joblib")

print("Model bundle saved to xgb_model.joblib")
