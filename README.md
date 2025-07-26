# FastAPI Triton Inference

A FastAPI application for text classification using Triton Inference Server.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Triton Server is running with the required model.

3. Run the application:
```bash
python main.py
```

The API will be available at http://localhost:8080

## API Endpoints

### POST /predict

Endpoint for text classification predictions.

Request body:
```json
{
    "text": "your text here"
}
```

Response:
```json
{
    "predictions": [array of probabilities]
}
```
