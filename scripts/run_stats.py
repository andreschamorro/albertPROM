#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import re
import random
import sys
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List
from functools import reduce
from collections import defaultdict, Counter
from functools import reduce
from multiprocessing import Pool

import datasets
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DATASET_TYPES = {"ngs": "loaders/ngs_script.py", "wtr": "loaders/trns_script.py"}
SPECIAL_DATASET_CONFIG = {
        "ngs": {'num_read': 0, 'x_fold': 5, 'len_r': 150, 'len_l': 150, 'std_dev': 50, 'dist': 500}}


def update_from_string(old: dict, update_str: str):
    d = dict(x.split("=") for x in update_str.split(","))
    for k, v in d.items():
        if k not in old:
            raise ValueError(f"key {k} isn't in the original config dict")
            
        old_v = old.get(k)
        if isinstance(old_v, bool):
            if v.lower() in ["true", "1", "y", "yes"]:
                v = True
            elif v.lower() in ["false", "0", "n", "no"]:
                v = False
            else:
                raise ValueError(f"can't derive true or false from {v} (key {k})")
        elif isinstance(old_v, int):
            v = int(v)
        elif isinstance(old_v, float):
            v = float(v)
        elif not isinstance(old_v, str):
            raise ValueError(
                f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
            )
        old[k] = v

def _kmer_split(k: int, sequence: str) -> List[str]:
        return [sequence[j: j + k] for j in range(len(sequence) - k + 1)]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    model_ksize: Optional[int] = field(
        default=8,
        metadata={"help": "K size"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + "cov, "},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The data dir of the dataset configuration."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    read_by_read: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    dataset_script_config = SPECIAL_DATASET_CONFIG.get(data_args.dataset_config_name, {})
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            DATASET_TYPES[data_args.dataset_name],
            data_args.dataset_config_name,
            data_dir=data_args.dataset_dir,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_script_config,
        )
    else:
        raise NotImplementedError
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    if data_args.dataset_name == "ngs":
        features_names = [col for col in column_names if col.startswith('read')]
    else:
        features_names = ["sequence"] if "sequence" in column_names else [column_names[0]]

    if data_args.dataset_name == "ngs":
        # TODO group for reads > max_length
        padding = "max_length" if data_args.pad_to_max_length else False
        if data_args.dataset_config_name.endswith('_single'):

            def tokenize_function(examples):
                kmer_example = [" ".join(kr) for kr in map(lambda r: _kmer_split(model_args.model_ksize, r), examples['read_1'])]
                return tokenizer(
                    kmer_example,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset single read",
                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                kmer_example = [f" {tokenizer.sep_token} ".join(
                    [" ".join(kr) for kr in map(lambda r: _kmer_split(model_args.model_ksize, r), z)])
                                for z in zip(*[examples[fn] for fn in features_names])]
                return tokenizer(
                    kmer_example,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every pairs in dataset",
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            kmer_example = [f" {tokenizer.sep_token} ".join([" ".join(kr) for kr in map(lambda r: _kmer_split(model_args.model_ksize, r), z)]) for z in zip(*[examples[fn] for fn in features_names])]
            return tokenizer(kmer_example, return_special_tokens_mask=True)
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every sequence in dataset",
            )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(tokenized_datasets)), min(3, len(tokenized_datasets))):
        logger.info(f"Sample {index} of the training set: {raw_datasets['train'][index][features_names[0]]}.")
        logger.info(f"Sample {index} of the training set: {tokenized_datasets['train'][index]}.")

    # Tokenizer stats
    if data_args.task_name == "cov":
        def countoken(l):
            return dict(Counter(list(map(lambda x: str(len(re.sub("[^actg\s]", "", x))), l))))
        def tokenize_stats(examples):
            stats = {}
            stats['ids_len'] = list(map(lambda e: len(e), examples['input_ids']))
            # stats['countoken'] = list(map(lambda e: countoken(tokenizer.convert_ids_to_tokens(e, skip_special_tokens=True)),
            #                               examples['input_ids']))
            stats['countunks'] = list(map(lambda e: e.count(tokenizer.unk_token_id), examples['input_ids']))
            return stats

        with training_args.main_process_first(desc="dataset map tokenization"):
            column_names = tokenized_datasets["train"].column_names
            stats_datasets = tokenized_datasets.map(
                tokenize_stats,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every sequence in dataset",
            )

        stats_datasets["train"].to_pandas().to_csv(os.path.join(training_args.output_dir, "stats_cov_{}.csv".format(model_args.model_ksize)))
    if data_args.task_name == "cou":
        def count_ids(examples):
            return dict(Counter(chain.from_iterable(examples)))
        def reduce_counts(count1, count2):
            """
            Combine (reduce) the passed two dictionaries to return
            a dictionary that contains the keys of both, where the
            values are equal to the sum of values for each key
            """
            # explicitly copy the dictionary, as otherwise
            # we risk modifying 'dict1'
            combined = defaultdict(lambda: 0)
            for key in count1:
                combined[key] = count1[key]

            for key in count2:
                combined[key] = count2[key]
            return dict(combined)

        def get_batch_corpus(data, batch_size):
            for i in range(0, len(data["train"]), batch_size):
                yield data["train"][i : i + batch_size]["input_ids"]

        results = map(count_ids, get_batch_corpus(tokenized_datasets, 1024*8))
        #with Pool() as pool:
        #    batch_size = len(tokenized_datasets["train"]) // data_args.preprocessing_num_workers
        #    results = pool.map_async(count_ids, list(get_batch_corpus(tokenized_datasets, batch_size)))

        total_count = reduce(reduce_counts, results)

        with open(os.path.join(training_args.output_dir, "stats_count_{}.json".format(model_args.model_ksize)), "w") as write_json:
            json.dump(total_count, write_json, indent=4)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
