import argparse
import glob
import os
import re
import io
from typing import List

import sentencepiece as spm

import datasets
from tokenizers import normalizers, pre_tokenizers, processors, SentencePieceUnigramTokenizer, Regex 

from transformers import (
    BertGenerationTokenizer,
    convert_slow_tokenizer,
    PreTrainedTokenizerFast
)

DATASET_TYPES = {"rds": "loaders/reads_script.py", "ngs": "loaders/ngs_script.py", "wtr": "loaders/trns_script.py"}

def create_lambda_with_globals(s):
    return eval(s, globals())

def _standardization(sequence):
    return re.sub(r'[^actg]', '', sequence.lower())

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def get_training_corpus(raw_datasets, features_names, batch_size, k):
    for feature in features_names:
        for seq in raw_datasets["train"][feature]:
            if len(seq)-k+1 >= 512:
                total_length = (len(seq)-k+1 // 512) * 512
                for i in range(0, total_length, 512):
                    yield _kmer_split(k, _standardization(seq[i:i+512]))
            else:
                yield _kmer_split(k, _standardization(seq))
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--name", default="sequencepiece_unigram", type=str, help="The name of the output vocab files"
    )
    parser.add_argument(
        "--dataset_name", default="wtr", type=str, help="The name of the input dataset"
    )
    parser.add_argument(
        "--dataset_config_name", default="transcript", type=str, help="The name of the input dataset"
    )
    parser.add_argument(
        "--dataset_dir", default="run/data", type=str, help="The name of the input dataset"
    )
    parser.add_argument(
        "--dataset_filter", default="lambda e: e", type=create_lambda_with_globals, help="A lambda filter of the input dataset"
    )
    parser.add_argument(
        "--vocab_size", default=10000, type=int, help="vocab size"
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int
    )
    parser.add_argument(
        "--k", default=17, type=int
    )
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    
    raw_datasets = datasets.load_dataset(DATASET_TYPES[args.dataset_name], args.dataset_config_name, data_dir=args.dataset_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    raw_datasets = raw_datasets.filter(args.dataset_filter)

    column_names = raw_datasets["train"].column_names
    if args.dataset_name == "ngs" or args.dataset_name == "rds" :
        features_names = [col for col in column_names if col.startswith('read')]
    else:
        features_names = ["sequence"] if "sequence" in column_names else [column_names[0]]

    # Initialize an empty tokenizer
    with io.BytesIO() as model:
        spm.SentencePieceTrainer.train(
                sentence_iterator=get_training_corpus(raw_datasets, features_names, args.batch_size, args.k), 
                model_writer=model, model_type="unigram", vocab_size=args.vocab_size, 
                max_sentencepiece_length=512, max_sentence_length=9216, train_extremely_large_corpus=True, num_threads=32)

        # Save the files
        # Serialize the model as file.
        with open(os.path.join(args.out, args.name+'.model'), 'wb') as f:
            f.write(model.getvalue())

if __name__ == "__main__":
    main()
