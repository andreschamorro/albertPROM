import os
import requests
import json
from io import StringIO
from typing import Optional, List
import pandas as pd
from Bio import SeqIO

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.head("/")
@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="static/index.html", media_type="text/html")

#pipe = pipeline("sentiment-analysis", "models/transcript", framework="pt", num_workers=16)

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def preprocess_function(reads_1, reads_2, k=15, sep_token="[SEP]"):
    # Tokenize the reads
    for r1, r2 in zip(reads_1, reads_2):
        yield f" {sep_token} ".join(
                [_kmer_split(k, str(r1.seq)), _kmer_split(k, str(r2.seq))])

class SequenceRequest(BaseModel):
    fformat: str
    reads_1: str
    reads_2: str

@app.post("/transcript", response_class = Response)
def predict(request: SequenceRequest):
    reads_1 = SeqIO.parse(StringIO(request.reads_1), request.fformat) 
    reads_2 = SeqIO.parse(StringIO(request.reads_2), request.fformat) 
    inference = pipe(preprocess_function(reads_1, reads_2))
    headers = {'Content-Disposition': 'attachment; filename="inference.csv"'}
    return Response(pd.DataFrame(inference).to_csv(), headers=headers, media_type="text/csv")
