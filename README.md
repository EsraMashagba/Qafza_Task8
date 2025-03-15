# Financial News Ranking System

This project analyzes financial news articles to assess their sentiment and impact on investment decisions. It classifies news into sentiment categories and ranks them based on investment importance using FinBERT.

## Features
✅ Classifies news sentiment using FinBERT
✅ Computes impact scores for financial news
✅ Ranks news articles based on sentiment and impact
✅ Implements MLOps practices with MLflow tracking
✅ Provides a FastAPI-based REST API for classification and ranking

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train and Save Model
Run the `model.py` script to preprocess data, train the model, and save it:
```bash
python model.py
```

### 2. Start the API Server
Run the `app.py` file to launch the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Test the API
#### Classify News
Send a POST request to classify financial news:
```bash
curl -X 'POST' \
  'http://localhost:5000/classify/' \
  -H 'Content-Type: application/json' \
  -d '{"news": "Stock market surges after positive earnings."}'
```
#### Rank Multiple News Articles
```bash
curl -X 'POST' \
  'http://localhost:5000/rank/' \
  -H 'Content-Type: application/json' \
  -d '[{"news": "Stock market surges after positive earnings."}, {"news": "Recession fears grow due to inflation."}]'
```

## Project Structure
```
│── model.py      # Loads dataset, preprocesses, trains model, saves FinBERT
│── ranking.py    # Implements ranking system based on sentiment & impact
│── app.py        # FastAPI-based web service for classification & ranking
│── requirements.txt  # Required dependencies
│── deploy.py       # Deploy model in hugging face


```

## MLOps Integration
- MLflow is used for experiment tracking.
- Models and logs are stored for future analysis.

## Authors
- **Esra'a Mashagba**



