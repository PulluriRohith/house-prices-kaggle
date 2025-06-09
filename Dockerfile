# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY app.py                    .
COPY preprocess_house_prices.py .
COPY xgb_model.joblib           .

EXPOSE 8001

# Cross-platform: Uvicorn binds to 0.0.0.0 for host traffic mapping
CMD ["uvicorn", "app:app", "--host", "0.0.0.0",   "--port", "8001"]