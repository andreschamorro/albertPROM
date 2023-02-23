import argparse
import glob
import os
from typing import List

import datasets
from tokenizers import normalizers, pre_tokenizers, processors, SentencePieceUnigramTokenizer, Regex 
from transformers import PreTrainedTokenizerFast

DATASET_TYPES = {"rds": "loaders/reads_script.py", "ngs": "loaders/ngs_script.py", "wtr": "loaders/trns_script.py"}

def create_lambda_with_globals(s):
    return eval(s, globals())

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def get_training_corpus(raw_datasets, features_names, batch_size, k):
    for i in range(0, len(raw_datasets["train"]), batch_size):
        for feature in features_names:
            yield [_kmer_split(k, seq) for seq in raw_datasets["train"][i : i + batch_size][feature]]

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
    tokenizer = SentencePieceUnigramTokenizer()

    tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Nmt(), normalizers.Lowercase(), normalizers.Replace(Regex("[^actg\s]"), "")]
            )
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        tokenizer.pre_tokenizer,])

    # And then train
    tokenizer.train_from_iterator(
        get_training_corpus(raw_datasets, features_names, args.batch_size, args.k),
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["[CLS]", "<pad>", "[SEP]", "<unk>", "[MASK]",],
        unk_token="<unk>",
    )

    tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
                ]
            )
    if args.fast:
        fast_tokenizer =  PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token='[CLS]', eos_token='[SEP]', 
                unk_token='<unk>', sep_token='[SEP]', 
                cls_token='[CLS]', pad_token='<pad>', mask_token='[MASK]',
                truncation_side='right')
        fast_tokenizer.save_pretrained(os.path.join(args.out, args.name))
    # Save the files
    else:
        tokenizer.save_model(args.out, args.name)

if __name__ == "__main__":
    main()
