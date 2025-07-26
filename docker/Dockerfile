FROM nvcr.io/nvidia/tritonserver:23.04-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPORT 8000

CMD ["tritonserver", "--model-repository=/models"]