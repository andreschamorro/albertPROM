#!/usr/bin/env python
"""
usage:
    $  ./make_kfold_paired.py PRESENCE_PREFIX ABSENCE_PREFIX SAMPLE_DIR N_SPLITS
"""
import sys, os, random
from heapq import nlargest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import StratifiedKFold

# Get inputs
p_prefix = sys.argv[1]
a_prefix = sys.argv[2]
d_prefix = sys.argv[3]
n_splits = int(sys.argv[4])

prid_processing = lambda i: i
abid_processing = lambda i: i.split('|')[0] + i.split('-')[-1]

all_features = []
all_labels = []
# Presence sample
print("presence processing ...")
pr_count = 0
with open(f"{p_prefix}1.fq", "r") as pr1_file, open(f"{p_prefix}1.fq", "r") as pr2_file:
    for r1, r2 in zip(SeqIO.parse(pr1_file, "fastq"), SeqIO.parse(pr2_file, "fastq")):
        r1.id = prid_processing(r1.id)
        r1.description="presence"
        r2.id = prid_processing(r2.id)
        r2.description="presence"
        all_features.append([r1, r2])
        all_labels.append(1)
        pr_count += 1

print(f"presence count: {pr_count}")
print("absence processing ...")

ab_count = 0
with open(f"{a_prefix}1.fq", "r") as ar1_file, open(f"{a_prefix}2.fq", "r") as ar2_file:
    for r1, r2 in zip(SeqIO.parse(ar1_file, "fastq"), SeqIO.parse(ar2_file, "fastq")):
        r1.id = abid_processing(r1.id)
        r1.description="absence"
        r2.id = abid_processing(r2.id)
        r2.description="absence"
        all_features.append([r1, r2])
        all_labels.append(0)
        ab_count += 1

print(f"absence count: {ab_count}")
# suffle data
print("All sample processed")
skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
print("Write sample ...")
try:
   os.makedirs(d_prefix)
except FileExistsError:
   # directory already exists
   pass

for k, (train_index, test_index) in enumerate(skf.split(all_features, all_labels)):
    train_fold_1 = [all_features[i][0] for i in train_index]
    train_fold_2 = [all_features[i][1] for i in train_index]
    # Train sample
    try:
       os.makedirs(os.path.join(d_prefix, "train"))
    except FileExistsError:
       # directory already exists
       pass
    with open(os.path.join(d_prefix, "train", f"{k}_R1.fq"), "w") as r1_file:
        SeqIO.write(train_fold_1, r1_file, "fastq")
    with open(os.path.join(d_prefix, "train", f"{k}_R2.fq"), "w") as r2_file:
        SeqIO.write(train_fold_2, r2_file, "fastq")
    # Test sample
    test_fold_1 = [all_features[i][0] for i in test_index]
    test_fold_2 = [all_features[i][1] for i in test_index]
    try:
       os.makedirs(os.path.join(d_prefix, "test"))
    except FileExistsError:
       # directory already exists
       pass
    with open(os.path.join(d_prefix, "test", f"{k}_R1.fq"), "w") as r1_file:
        SeqIO.write(test_fold_1, r1_file, "fastq")
    with open(os.path.join(d_prefix, "test", f"{k}_R2.fq"), "w") as r2_file:
        SeqIO.write(test_fold_2, r2_file, "fastq")

print("Done")
