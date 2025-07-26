# Speech & Text Classification Service

This project provides two FastAPI services:
1. Text Classification Service (`main.py`)
2. Speech Processing & Classification Service (`main2.py`)

## Service Overview

### Text Classification Service (main.py)
- Simple text classification using Triton Inference Server
- Endpoint: `/predict`
- Input: Text data
- Output: Classification predictions
- Port: 8080

### Speech Processing Service (main2.py)
- Complete speech processing pipeline with ASR and classification
- Endpoint: `/eval`
- Input: Audio file (.wav) and agent data
- Output: Transcription and classification results
- Port: 4000

## Project Structure
```
├── main.py                 # Text classification service
├── main2.py                # Speech processing service
├── test_asr.py             # Test script
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   └── requirements.txt
└── model_repository/       # Triton model configurations
    ├── my_model/          # ONNX model for classification
    └── sst/               # Python model for ASR
```

## Features

### Text Classification (main.py)
- Fast text processing
- Hugging Face tokenizer integration
- Simple JSON input/output

### Speech Processing (main2.py)
1. Audio Processing:
   - Voice Activity Detection (VAD)
   - Noise reduction
   - Bandpass filtering
   - Audio normalization
2. ASR (Automatic Speech Recognition)
   - Chunked processing with overlap
   - Text merging with overlap handling
3. Text Classification
   - 6 classification categories
   - Boolean results for each category

## Installation

1. Install dependencies:
```bash
pip install -r docker/requirements.txt
```

2. Start Triton Server:
```bash
tritonserver --model-repository=/path/to/model_repository
```

## Usage

### Text Classification (main.py)
1. Start the service:
```bash
python main.py
```

2. Make a request:
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "your text here"}'
```

### Speech Processing (main2.py)
1. Start the service:
```bash
python main2.py
```

2. Make a request:
```bash
curl -X POST "http://localhost:4000/eval" \
     -F "agent_data='{}'" \
     -F "voice_file=@path/to/audio.wav"
```

## API Documentation

### Text Classification API (main.py)

**Endpoint:** `/predict`

Request:
```json
{
    "text": "input text"
}
```

Response:
```json
{
    "predictions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}
```

### Speech Processing API (main2.py)

**Endpoint:** `/eval`

Request:
- `agent_data`: JSON string
- `voice_file`: WAV audio file

Response:
```json
{
    "transcription": "transcribed text",
    "is_greeting": true,
    "is_introself": false,
    "is_informlicense": true,
    "is_informobjective": false,
    "is_informbenefit": true,
    "is_informinterval": false
}
```

## Configuration

Environment variables in `main2.py`:
```python
SR_TARGET = 16000        # Target sample rate
CHUNK_MS = 30_000       # Chunk size for processing
OVERLAP_MS = 2_000      # Overlap between chunks
TRITON_URL = "localhost:8000"
ASR_MODEL_NAME = "sst"
CLS_MODEL_NAME = "my_model"
```

## Docker Support

Build and run with Docker:
```bash
cd docker
docker build -t triton-server .
docker run --gpus all -p 8000:8000 triton-server
```

## Development Notes

1. Audio Processing Pipeline:
   - VAD helps detect speech segments
   - Noise reduction improves ASR quality
   - Bandpass filter removes unwanted frequencies

2. Text Processing:
   - Uses Hugging Face tokenizer
   - Maximum sequence length: 128
   - Supports Thai language

3. Classification Categories:
   - Greeting detection
   - Self introduction
   - License information
   - Objective information
   - Benefit information
   - Interval information

## Troubleshooting

1. Audio Issues:
   - Ensure mono audio input
   - Check sample rate (16kHz recommended)
   - Verify WAV format

2. Server Issues:
   - Check if Triton Server is running
   - Verify model repository path
   - Check port availability

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.