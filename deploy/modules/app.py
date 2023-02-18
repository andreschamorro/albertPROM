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
from torch.utils.data import Dataset
from transformers import pipeline

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.head("/")
@app.get("/")
def index() -> FileResponse:
    return FileResponse(path="static/index.html", media_type="text/html")

pipe = pipeline("sentiment-analysis", "models/transcript", framework="pt", num_workers=16)

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def _read(fa, fformat):
    for feature in enumerate(SeqIO.parse(fa, fformat)) 
        yield feature 

def kmer_generator_pair(requests, k=15, sep_token="[SEP]"):
    # Tokenize the reads
    for r1, r2 in zip(_read(requests.reads_1, request.fformat), _read(requests.reads_2, request.fformat)):
        yield f" {sep_token} ".join(
                [_kmer_split(k, str(r1.seq)), _kmer_split(k, str(r2.seq))])

class SequenceRequest(BaseModel):
    fformat: str
    reads_1: str
    reads_2: str

@app.post("/transcript", response_class = Response)
def predict(request: SequenceRequest):
    try:
       os.makedirs('tmp')
       tmpdirname = 'tmp'
    except FileExistsError:
       # directory already exists
       pass
   #with tempfile.TemporaryDirectory() as tmpdirname:
    inference = pipe(kmer_generator_pair(request))
    line1_r1 = [r1 for r1, inf in zip(_read(requests.reads_1, request.fformat), inference)
                                if inf['label'] == 'presence']
    line1_r2 = [r2 for r2, inf in zip(_read(requests.reads_2, request.fformat), inference)
                                if inf['label'] == 'presence']
    # Write Temp line-1 reads
    with open(os.path.join(tmpdirname, f"line1s_R1.fq"), "w") as r1_file:
        SeqIO.write(line1_r1, r1_file, request.fformat)
    with open(os.path.join(tmpdirname, f"line1s_R2.fq"), "w") as r2_file:
            SeqIO.write(line1_r2, r2_file, request.fformat)
    headers = {'Content-Disposition': 'attachment; filename="inference.csv"'}
    return Response(pd.DataFrame(inference).to_csv(), headers=headers, media_type="text/csv")
