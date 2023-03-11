import os
import sys
import argparse
import random
import tempfile
import snakemake
import subprocess
import requests
import json
from tqdm import tqdm
from io import StringIO
from typing import Optional, List, Union
import numpy as np
import pandas as pd
from Bio import SeqIO
from itertools import repeat
import logging
from memory_profiler import profile
from datetime import datetime
import time

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

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)
from torch.quantization import quantize_dynamic_jit
from torch.jit import RecursiveScriptModule

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logging.getLogger("transformers.modeling_utils").setLevel(
        logging.WARN)  # Reduce logging

# Default memory profile file
mem_profile = sys.stdout


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device='cpu').view(shape).contiguous()

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

def _to_jit(requests, model):
    model.to(requests.device)

    input_ids = ids_tensor([8, requests.model_max_length], vocab_size=2)
    token_type_ids = ids_tensor([8, requests.model_max_length], vocab_size=2)
    attention_mask = ids_tensor([8, requests.model_max_length], vocab_size=2)
    dummy_input = (input_ids, attention_mask, token_type_ids)
    traced_model = torch.jit.trace(model, dummy_input)
    return traced_model

@profile(stream=mem_profile)
def _salmon(**kwargs):
    import snakemake
    snakefile = "deploy/resources/snakemake/snakefile.paired" if kwargs["paired"] else "deploy/resources/snakemake/snakefile.single"

    snakemake.snakemake(
        snakefile=snakefile,
        config={
            "input_path": kwargs["input_path"],
            "output_path": kwargs["--outpath"],
            "index": kwargs["index"],
            "salmon": os.path.join(os.path.expanduser('~'),".local/bin/salmon"),
            "num_threads" : kwargs["num_threads"],
            "exprtype": kwargs["exprtype"],
        },
        quiet=True,
        lock=False
    )

def _kmer_split(k: int, sequence: str) -> List[str]:
    return [sequence[j: j + k] for j in range(len(sequence) - k + 1)]

def _read(ffile, fformat):
    for feature in SeqIO.parse(ffile, fformat):
        yield feature 

@profile(stream=mem_profile)
def predict(request, model: RecursiveScriptModule, tokenizer):
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
    # Inference
    model = model.to(request.device)
    model.eval()
    # Predict
    ## Proprocessing
    read_1_key, read_2_key = "read_1", "read_2"
    def preprocess_function(examples):
        # Tokenize the reads
        kmer_example = [" ".join(
            [" ".join(kr) for kr in map(lambda r: _kmer_split(request.model_ksize, r), z)]) 
                        for z in zip(examples[read_1_key],  examples[read_2_key])]
        return tokenizer(kmer_example, padding=request.padding, max_length=request.max_seq_length, truncation=True)

    raw_datasets = datasets.load_dataset("loaders/fast_script.py", name="paired_fast", data_files={"read_1": request.reads_1, "read_2": request.reads_2})

    column_names = raw_datasets["train"].column_names
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=request.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    logger.info("***** Tokenization map done *****")
    request.eval_batch_size = request.per_gpu_eval_batch_size * max(1, request.n_gpu)
    raw_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    dataloader = DataLoader(raw_datasets['train'], batch_size=request.eval_batch_size)

    # multi-gpu eval
    if request.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running prediction {} *****".format(request.prefix))
    logger.info("  Num examples = %d", len(raw_datasets))
    logger.info("  Batch size = %d", request.eval_batch_size)
    time_start = time.clock()
    preds = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(request.device) for t in batch.values())
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            outputs = model(**inputs)
            logits = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    time_end = time.clock()
    logger.info("  Prediction elapsed time %.5f", time_end-time_start)
    # r1 and r2 has the same id
    line1_ids = [r1.id.encode() for r1, p in zip(_read(request.reads_1, request.fformat), preds) if p == 0]
    seq_grep_r1 = subprocess.Popen(['seqkit', 
                                 'grep', '-f', '-',
                                 request.reads_1, '-o', os.path.join(request.out, "presence_R1.fa")], 
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    seq_grep_r2 = subprocess.Popen(['seqkit', 
                                 'grep', '-f', '-',
                                 request.reads_2, '-o', os.path.join(request.out, "presence_R2.fa")], 
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    _ = seq_grep_r1.communicate(input=b'\n'.join(line1_ids))
    _ = seq_grep_r2.communicate(input=b'\n'.join(line1_ids))
    logger.info("***** Running Salom {} *****".format(request.prefix))
    time_start = time.clock()
    _salmon(**salmon_kargs)
    time_end = time.clock()
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
        "-p", "--prefix", default="", type=str
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

    tokenizer = AutoTokenizer.from_pretrained(
        "deploy/models/transcript",
        use_fast=True,
        model_max_length=512,
        truncation_side='right',
        use_auth_token=None,
    )

    if False: # model is not jit
        config = AutoConfig.from_pretrained("deploy/models/transcript", num_labels=2, finetuning_task="trc2", use_auth_token=None, return_dict=False)
        model = AutoModelForSequenceClassification.from_pretrained("deploy/models/transcript", config=config, use_auth_token=None)
        model = _to_jit(model)
        torch.jit.save(model, "deploy/models/TorchScript/traced_albert.pt")
    else:
        model= torch.jit.load("deploy/models/TorchScript/traced_albert.pt")
    predict(args, model, tokenizer) 

if __name__ == "__main__":
    main()
