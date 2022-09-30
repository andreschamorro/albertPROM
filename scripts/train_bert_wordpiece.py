import argparse
import glob
from typing import List

from tokenizers import BertWordPieceTokenizer

def _kmer_split(k: int, sequence: str) -> List[str]:
    return " ".join([sequence[j: j + k] for j in range(len(sequence) - k + 1)])

def get_training_corpus(trans_data, batch_size, k):
    for i in range(0, len(trans_data["train"]), batch_size):
        yield [_kmer_split(k, seq) for seq in trans_data["train"][i : i + batch_size]["feature"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--name", default="bert-wordpiece", type=str, help="The name of the output vocab files"
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
    args = parser.parse_args()
    
    files = glob.glob(args.files)
    if not files:
        print(f"File does not exist: {args.files}")
        exit(1)
    
    
    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
    
    trans_data = datasets.load_dataset('loaders/dataset_script.py', data_dir='run/data')
    trans_data = trans_data.shuffle(seed=42)

    # And then train
    tokenizer.train_from_iterator(
        get_training_corpus(trans_data, args.batch_size, args.k),
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=3000,
        wordpieces_prefix="##",
    )
    
    # Save the files
    tokenizer.save_model(args.out, args.name)

if __name__ == "__main__":
    main()
