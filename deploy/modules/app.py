import os
import requests
import json
from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(docs_url=None, redoc_url=None)
pipe = pipeline("sentiment-analysis", "models/transcript", framework="pt")

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def preprocess_function(read_1, read_2, k=15, sep_token="[SEP]"):
    # Tokenize the reads
    return f" {sep_token} ".join(
        [_kmer_split(k, read_1), _kmer_split(k, read_2)])

class SequenceRequest(BaseModel):
    read_1: str
    read_2: str

class SequenceResponse(BaseModel):
    label: str
    confidence: float

@app.post("/transcript", response_model=SequenceResponse)
def predict(request: SequenceRequest):
    inference = pipe(preprocess_function(request.read_1, request.read_2))
    return SequenceResponse(
        label=inference['label'], confidence=inference['score']
    )
