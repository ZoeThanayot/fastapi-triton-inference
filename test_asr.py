import numpy as np
import tritonclient.http as httpclient
import json

# Create client for Triton server
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# Create random input data
input_0 = np.random.randn(1, 128).astype(np.int64)
input_1 = np.random.randn(1, 128).astype(np.int64)

# Create triton inputs
inputs = [
    httpclient.InferInput("input_ids", input_0.shape, "INT64"),
    httpclient.InferInput("attention_mask", input_1.shape, "INT64"),
]

# Initialize inputs
inputs[0].set_data_from_numpy(input_0)
inputs[1].set_data_from_numpy(input_1)

# Create outputs
outputs = [httpclient.InferRequestedOutput("output")]

# Query Triton server
results = triton_client.infer(
    model_name="my_model",
    inputs=inputs,
    outputs=outputs
)

# Get output
output_data = results.as_numpy("output")
print(output_data)