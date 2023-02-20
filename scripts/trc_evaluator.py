import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Union

import datasets
import numpy as np
from sklearn.model_selection import KFold
from datasets import load_dataset

from ray.tune.schedulers import PopulationBasedTrainingReplay
from ray import air, tune

import evaluate
from evaluate import evaluator
import transformers
from transformers import pipeline
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

task_to_keys = {
    "trc1": ("read_1", None),
    "trc2": ("read_1", "read_2"),
}

task_to_labelkeys = {
    "trc1": ("label_1", None),
    "trc2": ("label_1", "label_2"),
}

DATASET_TYPES = {"rds": "loaders/reads_script.py", "ngs": "loaders/ngs_script.py", "wtr": "loaders/trns_script.py"}

def _kmer_split(k: int, sequence: str) -> List[str]:
    return [sequence[j: j + k] for j in range(len(sequence) - k + 1)]

def main():
    parser = HfArgumentParser((TrainingArguments,))
    training_args, _ = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("run/trc2_5x_trns_paired")

    task_name = "trc2"
    dataset_name = "rds"
    model_ksize = 15
    dataset_config_name = "paired_mono"
    dataset_dir = "run/data"
    preprocessing_num_workers = 48
    max_eval_samples = None
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        DATASET_TYPES[dataset_name],
        dataset_config_name,
        data_dir=dataset_dir,
        use_auth_token=True,
    ).shuffle()
    if "validation" not in raw_datasets.keys():
        logger.info("Not Validation dataset, split from train dataset")
        raw_datasets["validation"] = load_dataset(
            DATASET_TYPES[dataset_name],
            dataset_config_name,
            split=f"train[:20%]",
            data_dir=dataset_dir,
            use_auth_token=True,
        ).shuffle()
        raw_datasets["train"] = load_dataset(
            DATASET_TYPES[dataset_name],
            dataset_config_name,
            split=f"train[20%:]",
            data_dir=dataset_dir,
            use_auth_token=True,
        ).shuffle()
    if "test" not in raw_datasets.keys():
        logger.info("Not Testing dataset, split from validation dataset")
        raw_datasets["test"] = load_dataset(
            DATASET_TYPES[dataset_name],
            dataset_config_name,
            split=f"validation[:30%]",
            data_dir=dataset_dir,
            use_auth_token=True,
        )
        raw_datasets["validation"] = load_dataset(
            DATASET_TYPES[dataset_name],
            dataset_config_name,
            split=f"validation[30%:]",
            data_dir=dataset_dir,
            use_auth_token=True,
        )

    column_names = raw_datasets["train"].column_names

    if task_name == "trc1":
        label_list = raw_datasets["train"].features["label"].names
    else:
        label_list = raw_datasets["train"].features["label_1"].names
    num_labels = len(label_list)

    padding = "max_length"

    read_1_key, read_2_key = task_to_keys[task_name]
    label_1_key, label_2_key = task_to_labelkeys[task_name]

    if dataset_config_name.startswith('single'):
        def preprocess_function(examples):
            # Tokenize the reads
            kmer_example = {}
            kmer_example["text"] = [" ".join(kr) for kr in 
                            map(lambda r: kmer_split(model_ksize, r), examples[read_1_key])]
            # Map labels to ids
            kmer_example["label"] = examples[label_1_key]
            return kmer_example
    else:
        def preprocess_function(examples):
            # Tokenize the reads
            kmer_example = {}
            kmer_example["text"] = [f" {tokenizer.sep_token} ".join(
                [" ".join(kr) for kr in map(lambda r: _kmer_split(model_ksize, r), z)]) 
                            for z in zip(examples[read_1_key],  examples[read_2_key])]
            # Map labels to ids
            kmer_example["label"] = [l1*l2 if l1 != -1 and l2 != -1 else -1 for l1, l2 in zip(examples[label_1_key],  examples[label_2_key])]
            return kmer_example 

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = raw_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.shuffle().select(range(max_eval_samples))

    os.system('clear')
    task_evaluator = evaluator("text-classification")
    pipe = pipeline("sentiment-analysis", "run/trc2_5x_trns_paired", framework="pt", num_workers=32)
    
    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=eval_dataset,
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        label_mapping={"presence": 0, "absence": 1}
    )
    print(eval_results)

if __name__ == "__main__":
    main()
