import json
import numpy as np
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "/mnt/myssd/byzoe/miticlass_wangchan"
model = None
tokenizer = None

def initialize(args):
    global model, tokenizer
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def execute(requests):
    global model, tokenizer
    
    responses = []
    
    for request in requests:
        input_ids = pb_tensor_to_tensor(request.inputs[0])
        attention_mask = pb_tensor_to_tensor(request.inputs[1])
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        output_tensor = pb_tensor("output", probs.cpu().numpy())
        response = pb_response(output_tensor)
        responses.append(response)
    
    return responses

def pb_tensor(name, array):
    from triton.server.model_config_pb2 import ModelInput, ModelOutput
    from tritonclient.grpc import model_config_pb2
    
    return model_config_pb2.ModelInferResponse().InferOutputTensor(
        name=name, shape=array.shape, datatype=1, contents=array
    )

def pb_tensor_to_tensor(pb_tensor):
    return torch.tensor(pb_tensor.contents.int64_contents).view(*pb_tensor.shape)

def pb_response(output_tensor):
    from triton.server.model_config_pb2 import ModelInput, ModelOutput
    from tritonclient.grpc import model_config_pb2
    
    return model_config_pb2.ModelInferResponse(
        outputs=[output_tensor]
    )