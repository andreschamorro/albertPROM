#!/usr/bin/env python
# coding: utf-8
from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# importing module
import sys
# appending a path
sys.path.extend(['.', '..'])
import datasets

trans_data = datasets.load_dataset('loaders/dataset_script.py', data_dir='run/data')
trans_data = trans_data.shuffle(seed=42)
trans_data = trans_data.map(lambda seq: {"feature": seq["feature"].upper()}, num_proc=4)

k = 17

def kmernizer(seq):
    return " ".join([seq[i: i + k] for i in range(len(seq) - k + 1)])

def batch_kmer(batch):
    return [kmernizer(seq) for seq in batch]

trans_data = trans_data.map(lambda seq: {"feature": kmernizer(seq["feature"])}, num_proc=4)

batch_size = 1024

def get_training_corpus():
    for i in range(0, len(trans_data["train"]), batch_size):
        yield trans_data["train"][i : i + batch_size]["feature"]

tokenizer.train_from_iterator(
        get_training_corpus(),
        vocab_size=100000, 
        show_progress=True,
        min_frequency=2, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>",])
tokenizer.save_pretrained("run/transcripts_bpe_tokenizer")

