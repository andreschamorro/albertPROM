import os
import argparse
import tempfile
import snakemake
import subprocess
import requests
import json
from tqdm import tqdm
from io import StringIO
from typing import Optional, List
import pandas as pd
from Bio import SeqIO
from itertools import repeat
import snakemake
from memory_profiler import profile
from datetime import datetime
import time
import logging

from transformers import (
    pipeline,
    TextClassificationPipeline, 
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import datasets

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logging.getLogger("transformers.modeling_utils").setLevel(
        logging.WARN)  # Reduce logging

# Default memory profile file
mem_profile = sys.stdout

@profile(stream=mem_profile)
def _salmon(**kwargs):
    snakefile = "deploy/resources/snakemake/snakefile.paired" if kwargs["paired"] else "deploy/resources/snakemake/snakefile.single"

    snakemake.snakemake(
        snakefile=snakefile,
        config={
            "input_path": kwargs["input_path"],
            "output_path": kwargs["output_path"],
            "index": kwargs["index"],
            "salmon": os.path.join(os.path.expanduser('~'),".local/bin/salmon"),
            "num_threads" : kwargs["num_threads"],
            "exprtype": kwargs["exprtype"],
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

@profile(stream=mem_profile)
def predict(request, pipe):
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
        "output_path": os.path.join(request.out, "salmon_out"),
        "index": "deploy/resources/IntactL1ElementsFLI-L1Ens84.38",
        "num_threads": 16,
        "exprtype": "TPM",
        "paired": True,
    }
    ## Preprocessing
    raw_datasets = datasets.load_dataset("loaders/fast_script.py", name="paired_fast", data_files={"read_1": request.reads_1, "read_2": request.reads_2})

    logger.info("***** Running prediction {} *****".format(request.prefix))
    logger.info("  Num examples = %d", len(raw_datasets))
    logger.info("  Batch size = %d", request.eval_batch_size)
    time_start = time.time()
    inference = pipe(raw_datasets, padding="max_length", max_length=512, truncation=True)
    # TODO
    # r1 and r2 has the same id
    line1_ids = [r1.id for r1, inf in zip(_read(request.reads_1, request.fformat), tqdm(inference))
                                if inf['label'] == 'presence']
    time_end = time.time()
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
    logger.info("***** Running Salom {} *****".format(request.prefix))
    salmon_kargs["paired"] = True 
    time_start = time.time()
    _salmon(**salmon_kargs)
    time_end = time.time()
    logger.info("  Salmon elapsed time %.5f", time_end-time_start)

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
    parser.add_argument(
        "--model_ksize", default=15, type=int
    )
    parser.add_argument(
        "--preprocessing_num_workers", default=32, type=int
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int
    )
    parser.add_argument(
        "--padding", default="max_length", type=str
    )
    # Set the device, batch size, topology, and caching flags.
    parser.add_argument(
        "--device", default="cpu", type=str
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int
    )
    parser.add_argument(
        "--n_gpu", default=0, type=int
    )
    parser.add_argument(
        "--local_rank", default=1, type=int
    )
    parser.add_argument(
        "--overwrite_cache", default=False, type=bool
    )
    parser.add_argument(
        "--log_file", default=None, type=str
    )
    parser.add_argument(
        "--mem_profile", default=None, type=str
    )
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w+')
        # Setup memory profile
    if args.mem_profile:
        mem_profile = open(f"memory_profile_{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}",'w+')

    config = AutoConfig.from_pretrained(
        "deploy/models/transcript",
        num_labels=2,
        finetuning_task="trc2",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "deploy/models/transcript",
        use_fast=True,
        model_max_length=512,
        use_auth_token=None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "deploy/models/transcript",
        config=config,
        use_auth_token=None,
    )

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, batch_size=16)
    predict(args, pipe)
    
if __name__ == "__main__":
    main()
