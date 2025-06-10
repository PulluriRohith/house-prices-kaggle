# House Prices Predictor (Kaggle)

This project implements a full machine-learning pipeline to predict house prices using the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  
It covers data preprocessing, model training, evaluation, and FastAPI-based deployment for real-time inference.

---

## Project Structure

| Path / File | Description |
|-------------|-------------|
| **src/model.py** | Trains the XGBoost model (preprocessing, feature engineering, evaluation, artefact saving). |
| **src/app.py** | FastAPI service exposing prediction endpoints. |
| **src/preprocess_house_prices.py** | Reusable preprocessing pipeline (cleaning, encoding, scaling). |
| **data/** | Holds `train.csv`, `test.csv`, and the generated `submission.csv`. |
| **models/** | Stores the trained model and preprocessing artefacts (`xgb_model.joblib`). |
| **README.md** | Project documentation (this file). |

---

## Features

* Robust preprocessing & feature engineering  
* XGBoost regression with evaluation (MSE, R², RMSLE)  
* Final retraining on the full dataset for leaderboard submission  
* Two FastAPI endpoints:  
  * **/predict** – expects 9 key features (fills missing with medians)  
  * **/predict-raw** – accepts any subset of raw features  

---

## Installation & Quick Start

### 1 — Clone & install dependencies
```bash
git clone https://github.com/rohith-pulluri_sap/house-prices-kaggle.git
cd house-prices-kaggle
pip install -r requirements.txt
```

### 2 — Train the model & generate `submission.csv`
```bash
python src/model.py
```

### 3 — Launch the FastAPI server
```bash
python src/app.py
```
The API is then available at **http://localhost:8001**

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| **GET** | `/` | Health check |
| **POST** | `/predict` | Predict price from 9 key features |
| **POST** | `/predict-raw` | Predict price from any raw feature subset |

### Example payload for `/predict`
```json
{
  "OverallQual": 7,
  "GrLivArea": 1500,
  "GarageCars": 2,
  "TotalBsmtSF": 800,
  "FullBath": 2,
  "YearBuilt": 2005,
  "YearRemodAdd": 2010,
  "LotArea": 8500,
  "Fireplaces": 1
}
```

---

## Outputs

* **submission.csv** – ready-to-upload Kaggle predictions  
* **models/xgb_model.joblib** – trained model + preprocessing artefacts  

---

## Acknowledgments

* Kaggle House Prices dataset  
* Libraries: **pandas**, **numpy**, **xgboost**, **joblib**, **fastapi**, **uvicorn**
