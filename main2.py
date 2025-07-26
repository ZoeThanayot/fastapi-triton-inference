from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import numpy as np
import tritonclient.http as httpclient

app = FastAPI()

# โหลด tokenizer
model_name_or_path = "/mnt/myssd/byzoe/miticlass_wangchan"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# สร้าง client เชื่อม Triton Server (แก้ URL ตามจริง)
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextInput):
    # 1. tokenize ข้อความ
    encoded = tokenizer(
        data.text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]         # shape (1, 128)
    attention_mask = encoded["attention_mask"]  # shape (1, 128)

    # 2. สร้าง Triton Input
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]

    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)

    outputs = [httpclient.InferRequestedOutput("output")]

    # 3. เรียก Triton Server
    response = triton_client.infer(
        model_name="sst",
        inputs=inputs,
        outputs=outputs,
    )

    preds = response.as_numpy("output")  # shape (1,6)

    # 4. แปลงผลลัพธ์เป็น list และส่งกลับ
    return {"predictions": preds.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8080, reload=True)