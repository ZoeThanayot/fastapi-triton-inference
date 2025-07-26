from fastapi import FastAPI, File, Form, UploadFile
import json
import numpy as np
import tritonclient.http as httpclient
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from transformers import AutoTokenizer
import webrtcvad
import re
import unicodedata
import pandas as pd
import os
import tempfile

# ==== CONFIG ====
SR_TARGET = 16000
CHUNK_MS = 30_000
OVERLAP_MS = 2_000
TRITON_URL = "localhost:8000"
ASR_MODEL_NAME = "sst"
CLS_MODEL_NAME = "my_model"
TOKENIZER_PATH = "/mnt/myssd/byzoe/miticlass_wangchan"

# === LOAD ===
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

# === Helper Functions ===
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.55 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def numpy_to_pydub(audio_np, sample_rate):
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return AudioSegment(audio_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

def pydub_to_numpy(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    return samples / 32767.0

def split_audio_with_overlap(audio_np, sr, chunk_ms=10000, overlap_ms=1000):
    chunk_samples = int(sr * chunk_ms / 1000)
    overlap_samples = int(sr * overlap_ms / 1000)
    step = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while start < len(audio_np):
        end = start + chunk_samples
        chunk = audio_np[start:end]
        chunks.append(chunk)
        if end >= len(audio_np):
            break
        start += step
    return chunks

def merge_with_overlap(prev_text, new_text, overlap=10):
    for i in range(overlap, 0, -1):
        if prev_text.endswith(new_text[:i]):
            return prev_text + new_text[i:]
    return prev_text + new_text

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            if is_speech:
                triggered = True
                voiced_frames.append(frame)
        else:
            voiced_frames.append(frame)
            if not is_speech:
                triggered = False
    return b''.join(voiced_frames)

# === ASR Pipeline ===
def asr_pipeline(audio_path):
    waveform, sr = librosa.load(audio_path, sr=SR_TARGET)
    if waveform.ndim != 1:
        raise Exception("Audio must be mono.")

    # --- VAD ก่อน Denoise ---
    vad_sr = 16000
    if sr != vad_sr:
        data_16k = librosa.resample(waveform, orig_sr=sr, target_sr=vad_sr)
    else:
        data_16k = waveform

    pcm_data = (data_16k * 32767).astype(np.int16).tobytes()
    vad = webrtcvad.Vad(2)  # aggressiveness 0-3
    frame_duration = 30  # ms
    frames = []
    n = int(vad_sr * frame_duration / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(pcm_data):
        frames.append(pcm_data[offset:offset + n])
        offset += n

    voiced_audio_bytes = vad_collector(vad_sr, frame_duration, 300, vad, frames)
    voiced_audio = np.frombuffer(voiced_audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # ถ้า resample กลับเป็น sr เดิม
    if sr != vad_sr and len(voiced_audio) > 0:
        voiced_audio = librosa.resample(voiced_audio, orig_sr=vad_sr, target_sr=sr)
    elif len(voiced_audio) == 0:
        voiced_audio = waveform  # ถ้าไม่มีเสียงพูดเลย

    # --- Denoise ต่อ ---
    noise_sample = voiced_audio[:int(sr * 0.5)]
    denoised = nr.reduce_noise(y=voiced_audio, y_noise=noise_sample, sr=sr, prop_decrease=1.0)
    filtered = apply_bandpass_filter(denoised, 80, 3000, sr)
    audio_segment = numpy_to_pydub(filtered, sr)
    boosted = audio_segment + 15  # dB
    boosted_np = pydub_to_numpy(boosted)
    boosted_np /= max(1.0, np.max(np.abs(boosted_np)))  # normalize

    chunks = split_audio_with_overlap(boosted_np, sr=sr, chunk_ms=CHUNK_MS, overlap_ms=OVERLAP_MS)
    full_text = ""
    for chunk in chunks:
        input_tensor = httpclient.InferInput("AUDIO_INPUT", chunk.shape, "FP32")
        input_tensor.set_data_from_numpy(chunk.astype(np.float32))
        output = httpclient.InferRequestedOutput("TEXT_OUTPUT")
        response = triton_client.infer(
            model_name=ASR_MODEL_NAME,
            inputs=[input_tensor],
            outputs=[output],
        )
        text = response.as_numpy("TEXT_OUTPUT")[0].decode("utf-8").strip()
        if full_text:
            full_text = merge_with_overlap(full_text, text)
        else:
            full_text = text
    return full_text.replace(" ", "").strip()  # == raw (do not postprocess)

# === Classification Pipeline ===
def classify_text(text):
    encoded = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    outputs = [httpclient.InferRequestedOutput("output")]
    response = triton_client.infer(
        model_name=CLS_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
    preds = response.as_numpy("output")[0]  # shape (6,)

    keys = [
        "is_greeting",
        "is_introself",
        "is_informlicense",
        "is_informobjective",
        "is_informbenefit",
        "is_informinterval",
    ]
    bools = [bool(p >= 0.5) for p in preds]
    return dict(zip(keys, bools))

# === FastAPI ===
app = FastAPI()

@app.post("/eval")
async def eval(
    agent_data: str = Form(...),
    voice_file: UploadFile = File(...)
):
    # 1. แปลง agent_data (json str) เป็น dict
    agent = json.loads(agent_data)
    # 2. Save audio ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await voice_file.read())
        tmp_path = tmp.name
    try:
        # 3. ASR
        transcription = asr_pipeline(tmp_path)  # อย่า postprocess!
        # 4. Classification
        result = classify_text(transcription)
    finally:
        os.remove(tmp_path)
    # 5. คืนผลตามสเปก
    response = {
        "transcription": transcription,
        **result
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=4000, reload=True)