# app.py
# -----------------------------------------------
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from typing import Any
from preprocess_house_prices import preprocess_house_prices

app = FastAPI()

# Load the saved model bundle (model + stats + feat_cols + medians)
bundle     = joblib.load("../models/xgb_model.joblib")
model      = bundle["model"]
stats      = bundle["stats"]
feat_cols  = bundle["columns"]
MEDIANS    = bundle["medians"]   # now embedded in the bundle

# Full list of raw columns your preprocess expects
ALL_RAW_COLS = [
  "Id","MSSubClass","LotFrontage","LotArea","Street","Alley","LotShape",
  "LandContour","Utilities","LotConfig","LandSlope","Neighborhood",
  "Condition1","Condition2","BldgType","HouseStyle","OverallQual",
  "OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl",
  "Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual",
  "ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure",
  "BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF",
  "TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical",
  "1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath",
  "BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr",
  "KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu",
  "GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea",
  "GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF",
  "EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence",
  "MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition"
]

def _run_pipeline(df_raw: pd.DataFrame) -> float:
    # 1. Replace None → NaN for all columns
    df_raw = df_raw.replace({None: np.nan})
    # 2. Proceed with your existing preprocess → one-hot → align → predict
    df_proc, _    = preprocess_house_prices(df_raw, stats=stats)
    df_onehot     = pd.get_dummies(df_proc, dtype=float)
    df_aligned    = df_onehot.reindex(columns=feat_cols, fill_value=0)
    pred_log      = model.predict(df_aligned)[0]
    return float(np.expm1(pred_log))

@app.get("/")
async def read_root():
    return {"message": "House-Prices Predictor is up and running!"}

@app.post("/predict")
async def predict(payload: dict[str, Any]):
    """
    Predict using only the 9 important features.
    Missing important keys get filled with the training median;
    all other features become None.
    """
    row = {}
    # Reject empty payload outright

    if not payload:
        raise HTTPException(status_code=400, detail="Empty payload — please supply at least provide id.")

    # Build a complete raw row:
    row = {}
    for col, median in MEDIANS.items():
        row[col] = payload.get(col, median)
    # fill the rest of the raw columns with None
    for col in ALL_RAW_COLS:
        if col not in row:
            row[col] = None

    try:
        df_raw = pd.DataFrame([row])
        price  = _run_pipeline(df_raw)
        return {"sale_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

@app.post("/predict-raw")
async def predict_raw(payload: dict[str, Any]):
    """
    Predict using ANY subset of raw features.
    Missing keys become None.
    """
    row = {col: payload.get(col, None) for col in ALL_RAW_COLS}

    # Reject completely empty payload
    if not payload:
        raise HTTPException(status_code=400,detail="Empty payload — please supply at least one raw feature.")

    #Build a complete raw row, filling missing keys with None
    row = {col: payload.get(col, None) for col in ALL_RAW_COLS}

    try:
        df_raw = pd.DataFrame([row])
        price  = _run_pipeline(df_raw)
        return {"sale_price": round(price, 2)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
