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
Fine-tuning the library models for transposable reads classifications (BERT, ALBERT, RoBERTa...) on a NGS dataset.

"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Union
import numpy as np

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

import datasets
from datasets import load_dataset
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

sys.path.extend(['.'])
from model.model import AlbertForWeightedSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
DATASET_TYPES = {"rds": "loaders/reads_script.py", "ngs": "loaders/ngs_script.py", "wtr": "loaders/trns_script.py"}

task_to_keys = {
    "trc1": ("read_1", None),
    "trc2": ("read_1", "read_2"),
}

task_to_labelkeys = {
    "trc1": ("label_1", None),
    "trc2": ("label_1", "label_2"),
}

task_to_glue = {
    "trc1": "sst2",
    "trc2": "sst2",
}


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

def batch_split(k: int, batch: List[str]) -> List[str]:
    return list(map(lambda r: _kmer_split(k, r), batch))

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
    model_ksize: Optional[int] = field(
        default=8,
        metadata={"help": "K size"},
    )
    model_loss_weight: Optional[List[Union[int, float]]] = field(
        default_factory = lambda: None,
        metadata={"help": "A manual rescaling weight given to the loss of each batch element"})
    n_trials: Optional[int] = field(
        default=64, 
        metadata={"help": "trials number"},
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
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
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
    test_split_percentage: Optional[int] = field(
        default=50,
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
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is None and self.train_file is None and self.validation_file is None:
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
    # send_example_telemetry("run_mlm", model_args, data_args)

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
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            DATASET_TYPES[data_args.dataset_name],
            data_args.dataset_config_name,
            data_dir=data_args.dataset_dir,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        ).shuffle()
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                DATASET_TYPES[data_args.dataset_name],
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                data_dir=data_args.dataset_dir,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            ).shuffle()
            raw_datasets["train"] = load_dataset(
                DATASET_TYPES[data_args.dataset_name],
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                data_dir=data_args.dataset_dir,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            ).shuffle()
        if "test" not in raw_datasets.keys():
            raw_datasets["test"] = load_dataset(
                DATASET_TYPES[data_args.dataset_name],
                data_args.dataset_config_name,
                split=f"validation[:{data_args.test_split_percentage}%]",
                data_dir=data_args.dataset_dir,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["validation"] = load_dataset(
                DATASET_TYPES[data_args.dataset_name],
                data_args.dataset_config_name,
                split=f"validation[{data_args.test_split_percentage}%:]",
                data_dir=data_args.dataset_dir,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        raise NotImplementedError
    # Labels
    if data_args.task_name is not None:
        if data_args.task_name == "trc1":
            label_list = raw_datasets["train"].features["label"].names
        elif not data_args.dataset_config_name.startswith('paired'):
            label_list = raw_datasets["train"].features["label"].names
        else:
            label_list = raw_datasets["train"].features["label_1"].names
        num_labels = len(label_list)
    else:
        raise NotImplementedError
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

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
    def get_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            )
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
        return model

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    if "label" in column_names: column_names.remove("label")
    if data_args.task_name is not None:
        read_1_key, read_2_key = task_to_keys[data_args.task_name]
        label_1_key, label_2_key = task_to_labelkeys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if "label" not in name]
        if "read_1" in non_label_column_names and "read_2" in non_label_column_names:
            read_1_key, read_2_key = "read_1", "read_2"
        else:
            if len(non_label_column_names) >= 2:
                read_1_key, read_2_key = non_label_column_names[:2]
            else:
                read_1_key, read_2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if data_args.task_name is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.dataset_config_name.startswith('single'):
        def preprocess_function(examples):
            # Tokenize the reads
            kmer_example = [" ".join(kr) for kr in 
                            map(lambda r: kmer_split(model_args.model_ksize, r), examples[read_1_key])]
            result = tokenizer(kmer_example, padding=padding, max_length=max_seq_length, truncation=True)
            # Map labels to ids
            return result
    elif data_args.dataset_config_name.startswith('multi'):
        def preprocess_function(examples):
            # Tokenize the reads
            kmer_example = [f" ".join(
                [" ".join(kr) for kr in map(lambda r: _kmer_split(model_args.model_ksize, r), z)]) 
                            for z in zip(examples[read_1_key],  examples[read_2_key])]
            result = tokenizer(kmer_example, padding=padding, max_length=max_seq_length, truncation=True)
            return result
    else:
        def preprocess_function(examples):
            # Tokenize the reads
            kmer_example = [f" ".join(
                [" ".join(kr) for kr in map(lambda r: _kmer_split(model_args.model_ksize, r), z)]) 
                            for z in zip(examples[read_1_key],  examples[read_2_key])]
            #args = (
            #        (batch_split(model_args.model_ksize, examples[read_1_key]),) if read_2_key is None else (batch_split(model_args.model_ksize, examples[read_1_key]), batch_split(model_args.model_ksize, examples[read_2_key]))
            #)
            result = tokenizer(kmer_example, padding=padding, max_length=max_seq_length, truncation=True)
            # Map labels to ids
            result["label"] = [l1*l2 if l1 != -1 and l2 != -1 else -1 for l1, l2 in zip(examples[label_1_key],  examples[label_2_key])]
            return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), k=max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.shuffle().select(range(max_eval_samples))

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
    
        tune_config = {
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,
                "num_train_epochs": tune.choice([5, 10]),
                "max_steps": -1,  # Used for smoke test.
        }

        scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="eval_accuracy",
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                    "weight_decay": tune.uniform(0.0, 0.3),
                    "learning_rate": tune.uniform(1e-5, 5e-4),
                    "per_device_train_batch_size": [8, 16],
                    },
        )

        reporter = CLIReporter(
                parameter_columns={
                    "weight_decay": "w_decay",
                    "learning_rate": "lr",
                    "per_device_train_batch_size": "train_bs/cpu",
                    "num_train_epochs": "num_epochs",
                },
                metric_columns=["eval_accuracy", "eval_recall", "eval_f1", "epoch", "time_since_restore"],
        )

        best_run = trainer.hyperparameter_search(
                hp_space=lambda _: tune_config,
                backend="ray",
                n_trials=model_args.n_trials,
                resources_per_trial={"cpu": 1, "gpu": 0},
                scheduler=scheduler,
                keep_checkpoints_num=1,
                checkpoint_score_attr="training_iteration",
                stop={"eval_accuracy": 0.96},
                progress_reporter=reporter,
                local_dir="~/ray_results/",
                name="tune_transformer_trc",
                log_to_file=True,
        )
        print("Best trial id: {}".format(best_run.run_id))
        print("Best trial hyperparameter: {}".format(best_run.hyperparameters))
        print("Best objetive that was obtained for this run: {}".format(best_run.objective))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "sst2"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
             'Use "auto" for cluster. '
             "Defaults to None for local.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )

    args, _ = parser.parse_known_args()

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(args.ray_address)

    main()
