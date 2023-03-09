import os
import argparse
import tempfile
import snakemake
import subprocess
import requests
import json
from io import StringIO
from typing import Optional, List
import pandas as pd
from Bio import SeqIO
from itertools import repeat
from transformers import pipeline

pipe = pipeline("sentiment-analysis", "deploy/models/transcript", framework="pt", num_workers=16)

def _salmon(**kwargs):
    import snakemake
    snakefile = os.path.join(os.path.dirname(__file__), "snakemake/snakefile.paired" if kwargs["paired"] else "snakemake/snakefile.single")

    snakemake.snakemake(
        snakefile=snakefile,
        config={
            "input_path": kwargs["inpath"],
            "output_path": kwargs["--outpath"],
            "index": kwargs["--reference"],
            "salmon": os.path.join(os.path.expanduser('~'),".local/bin/salmon"),
            "num_threads" : kwargs["--num_threads"],
            "exprtype": kwargs["--exprtype"],
        },
        quiet=True,
        lock=False
    )

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def _read(ffile, fformat):
    for feature in SeqIO.parse(ffile, fformat):
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
                [_kmer_split(k, str(r1.seq)), r2])

def predict(request):
    try:
       os.makedirs(request.out)
    except FileExistsError:
       # directory already exists
       pass
    try:
       os.makedirs(os.path.join(request.out, "salmon_out"))
    except FileExistsError:
       # directory already exists
       pass
    salmon_kargs = {
        "input_path": request.out,
        "outpath": os.path.join(request.out, "salmon_out"),
        "index": "deploy/resource/IntactL1ElementsFLI-L1Ens84.38",
        "num_threads": 16,
        "exprtype": "TPM"
    }
    # Single reads
    if request.reads_2 == "":
        inference = pipe(kmer_generator_single(request))
        line1_ids = [r.id for r, inf in zip(_read(request.reads_1, request.fformat), inference)
                                    if inf['label'] == 'presence']
        with open(os.path.join(request.out, "presence_ids.list"), 'w') as pre_ids:
            pre_ids.write('\n'.join(line1_ids))
        seq_grep = subprocess.Popen(['seqkit', 
                                     'grep', '-f', os.path.join(request.out, "presence_ids.list"),
                                     request.reads_1, '-o', os.path.join(request.out, "presence_R1.fa")], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        _ = seq_grep.communicate()
        salmon_kargs["paired"] = False
        _salmon(salmon_kargs)
        # Write Temp line-1 reads
    else:
        inference = pipe(kmer_generator_single(request))
        # TODO
        # r1 and r2 has the same id
        line1_ids = [r1.id for r1, inf in zip(_read(request.reads_1, request.fformat), inference)
                                    if inf['label'] == 'presence']
        with open(os.path.join(request.out, "presence_ids.list"), 'w') as pre_ids:
            pre_ids.write('\n'.join(line1_ids))
        seq_grep_r1 = subprocess.Popen(['seqkit', 
                                     'grep', '-f', os.path.join(request.out, "presence_ids.list"),
                                     request.reads_1, '-o', os.path.join(request.out, "presence_R1.fa")], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        seq_grep_r2 = subprocess.Popen(['seqkit', 
                                     'grep', '-f', os.path.join(request.out, "presence_ids.list"),
                                     request.reads_2, '-o', os.path.join(request.out, "presence_R2.fa")], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        _ = seq_grep_r1.communicate()
        _ = seq_grep_r2.communicate()
        salmon_kargs["paired"] = True 
        _salmon(salmon_kargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "-1", "--reads_1", default="", type=str
    )
    parser.add_argument(
        "-2", "--reads_2", default="", type=str
    )
    parser.add_argument(
        "--fformat", default="fastq", type=str
    )
    args = parser.parse_args()

    predict(args)
    
if __name__ == "__main__":
    main()
