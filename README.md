# FastAPI Triton Inference Service

A FastAPI-based web service that provides text classification using NVIDIA Triton Inference Server. This project demonstrates how to serve multiple machine learning models using Triton Server with both ONNX and Python backends.

## Project Structure

```
├── docker/
│   ├── Dockerfile           # Docker configuration for Triton server
│   └── requirements.txt     # Python dependencies
├── model_repository/
│   ├── my_model/           # ONNX model configuration
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.onnx
│   └── sst/               # Python backend model
│       ├── config.pbtxt
│       └── 1/
│           └── model.py
├── main.py                # FastAPI service for my_model
├── main2.py               # FastAPI service for sst model
└── test_asr.py            # Test script for model inference
```

## Features

- FastAPI web service with async endpoints
- Support for multiple model backends (ONNX and Python)
- Hugging Face Transformers integration
- Docker containerization
- GPU acceleration support

## Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, for GPU acceleration)
- Docker (optional, for containerization)
- NVIDIA Container Toolkit (if using GPU with Docker)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ZoeThanayot/fastapi-triton-inference.git
cd fastapi-triton-inference
```

2. Install dependencies:
```bash
pip install -r docker/requirements.txt
```

3. Set up Triton Server:

   Using Docker:
   ```bash
   docker build -t triton-server docker/
   docker run --gpus all -p 8000:8000 triton-server
   ```

   Or install locally:
   ```bash
   apt-get install triton-server
   tritonserver --model-repository=/path/to/model_repository
   ```

## Running the Service

1. Start the FastAPI service:
```bash
# For my_model
python main.py

# For sst model
python main2.py
```

The service will be available at `http://localhost:8080`

## API Endpoints

### POST /predict

Makes a prediction on the input text.

**Request Body:**
```json
{
    "text": "your text here"
}
```

**Response:**
```json
{
    "predictions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}
```

## Model Configuration

### my_model (ONNX)
- Input shape: [1, 128]
- Output shape: [1, 6]
- Data type: INT64 (input), FP32 (output)
- GPU acceleration enabled

### sst (Python Backend)
- Input shape: [1, 128]
- Output shape: [1, 6]
- Uses Hugging Face Transformers
- CPU execution

## Testing

Use the test script to verify the model inference:
```bash
python test_asr.py
```

## Docker Support

The included Dockerfile sets up Triton Server with all required dependencies. Build and run:

```bash
cd docker
docker build -t triton-server .
docker run --gpus all -p 8000:8000 -v /path/to/model_repository:/models triton-server
```

## Environment Variables

- `MODEL_PATH`: Path to the model files (default: `/mnt/myssd/byzoe/miticlass_wangchan`)
- `TRITON_SERVER_URL`: Triton server URL (default: `localhost:8000`)
- `PORT`: FastAPI service port (default: 8080)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.