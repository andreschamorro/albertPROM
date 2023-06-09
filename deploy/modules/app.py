import os
import subprocess
import requests
import json
from io import StringIO
from typing import Optional, List
import pandas as pd
from Bio import SeqIO
from itertools import repeat

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
    for feature in SeqIO.parse(StringIO(fa), fformat):
        yield feature 

def kmer_generator_pair(request, k=15, sep_token=""):
    # Tokenize the reads
    for r1, r2 in zip(_read(request.reads_1, request.fformat), _read(request.reads_2, request.fformat)):
        yield f" {sep_token}".join(
                [_kmer_split(k, str(r1.seq)), _kmer_split(k, str(r2.seq))])

def kmer_generator_single(request, k=15, sep_token=""):
    # Tokenize the reads
    for r1, r2 in zip(_read(request.reads_1, request.fformat), repeat("")):
        yield f" ".join(
                [_kmer_split(k, str(r1.seq)), repeat("")])

class SequenceRequest(BaseModel):
    fformat: str
    reads_1: str
    reads_2: str

@app.post("/transcript", response_class = Response)
def predict(request: SequenceRequest):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
           os.makedirs(os.path.join(tmpdirname, "salmon_out"))
        except FileExistsError:
           # directory already exists
           pass
        salmon_kargs = {
            "input_path": tmpdirname,
            "outpath": os.path.join(tmpdirname, "salmon_out"),
            "index": "resource/IntactL1ElementsFLI-L1Ens84.38",
            "num_threads": 16,
            "exprtype": "TPM"
        }
        # Single reads
        if request.reads_2 == "":
            inference = pipe(kmer_generator_single(request))
            line1_ids = [r.id for r, inf in zip(_read(request.reads_1, request.fformat), inference)
                                        if inf['label'] == 'presence']
            with open(os.path.join(tmpdirname, "presence_ids.list"), 'w') as pre_ids:
                pre_ids.write('\n'.join(line1_ids))
            seq_grep = subprocess.Popen(['seqkit', 
                                         'grep', '-f', os.path.join(tmpdirname, "presence_ids.list"),
                                         '-', '-o', os.path.join(tmpdirname, "presence_R1.fa")], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            _ = seq_grep.communicate(input=request.reads_1.encode())
            salmon_kargs["paired"] = False
            # Write Temp line-1 reads
        else:
            inference = pipe(kmer_generator_single(request))
            # TODO
            # r1 and r2 has the same id
            line1_ids = [r1.id for r1, inf in zip(_read(request.reads_1, request.fformat), inference)
                                        if inf['label'] == 'presence']
            with open(os.path.join(tmpdirname, "presence_ids.list"), 'w') as pre_ids:
                pre_ids.write('\n'.join(line1_ids))
            seq_grep_r1 = subprocess.Popen(['seqkit', 
                                         'grep', '-f', os.path.join(tmpdirname, "presence_ids.list"),
                                         '-', '-o', os.path.join(tmpdirname, "presence_R1.fa")], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            seq_grep_r2 = subprocess.Popen(['seqkit', 
                                         'grep', '-f', os.path.join(tmpdirname, "presence_ids.list"),
                                         '-', '-o', os.path.join(tmpdirname, "presence_R2.fa")], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            _ = seq_grep_r1.communicate(input=request.reads_1.encode())
            _ = seq_grep_r2.communicate(input=request.reads_2.encode())
            salmon_kargs["paired"] = True 
    headers = {'Content-Disposition': 'attachment; filename="inference.csv"'}
    return Response(pd.DataFrame(inference).to_csv(), headers=headers, media_type="text/csv")
