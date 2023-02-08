#!/usr/bin/env python
"""
usage:
    $  ./make_subsample_paired.py PRESENCE_PREFIX ABSENCE_PREFIX SAMPLE_DIR N_SAMPLE RATIO VALSIZE 
"""
import sys, os, random
from heapq import nlargest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split

# Get inputs
p_prefix = sys.argv[1]
a_prefix = sys.argv[2]
d_prefix = sys.argv[3]
n_sample = int(sys.argv[4])
ratio = int(sys.argv[5])
val_size = int(sys.argv[6])

prid_processing = lambda i: i.split('|')[0] + i.split('-')[-1]
abid_processing = lambda i: i.split('|')[0] + i.split('-')[-1]

all_features = []
# Presence sample
print("presence processing ...")
p_features = []
pr_count = 0
with open(f"{p_prefix}1.fq", "r") as pr1_file, open(f"{p_prefix}1.fq", "r") as pr2_file:
    for r1, r2 in zip(SeqIO.parse(pr1_file, "fastq"), SeqIO.parse(pr2_file, "fastq")):
        r1.id = prid_processing(r1.id)
        r1.description="presence"
        r2.id = prid_processing(r2.id)
        r2.description="presence"
        p_features.append([r1, r2])
        pr_count += 1

print(f"presence count: {pr_count}")
n_sample = n_sample if (n_sample < pr_count and n_sample > 0) else pr_count
r_index = random.sample(range(pr_count), n_sample)
all_features = [p_features[i] for i in r_index]
n_sample = n_sample * ratio if n_sample > 0 else pr_count * ratio
print("absence processing ...")
a_features = []
ab_count = 0
with open(f"{a_prefix}1.fq", "r") as ar1_file:
    for r1 in SeqIO.parse(ar1_file, "fastq"):
        ab_count += 1
print(f"absence count: {ab_count}")

r_index = random.sample(range(ab_count), n_sample if n_sample < ab_count else ab_count)
with open(f"{a_prefix}1.fq", "r") as ar1_file, open(f"{a_prefix}2.fq", "r") as ar2_file:
    for i, (r1, r2) in enumerate(zip(SeqIO.parse(ar1_file, "fastq"), SeqIO.parse(ar2_file, "fastq"))):
        if i in r_index:
            r1.id = abid_processing(r1.id)
            r1.description="absence"
            r2.id = abid_processing(r2.id)
            r2.description="absence"
            all_features.append([r1, r2])

print("All sample processed")
# suffle data
print("Write sample ...")

train_fold, test_fold = train_test_split(all_features, random_state=1, shuffle=True)
test_fold, val_fold = train_test_split(test_fold, test_size=val_size, random_state=1)
# Train sample
try:
   os.makedirs(d_prefix)
except FileExistsError:
   # directory already exists
   pass
with open(os.path.join(d_prefix, f"train_paired_R1.fq"), "w") as r1_file:
    SeqIO.write([train_fold[i][0] for i in range(len(train_fold))], r1_file, "fastq")
with open(os.path.join(d_prefix, f"train_paired_R2.fq"), "w") as r2_file:
    SeqIO.write([train_fold[i][1] for i in range(len(train_fold))], r2_file, "fastq")

with open(os.path.join(d_prefix, f"test_paired_R1.fq"), "w") as r1_file:
    SeqIO.write([test_fold[i][0] for i in range(len(test_fold))], r1_file, "fastq")
with open(os.path.join(d_prefix, f"test_paired_R2.fq"), "w") as r2_file:
    SeqIO.write([test_fold[i][1] for i in range(len(test_fold))], r2_file, "fastq")

with open(os.path.join(d_prefix, f"val_paired_R1.fq"), "w") as r1_file:
    SeqIO.write([val_fold[i][0] for i in range(len(val_fold))], r1_file, "fastq")
with open(os.path.join(d_prefix, f"val_paired_R2.fq"), "w") as r2_file:
    SeqIO.write([val_fold[i][1] for i in range(len(val_fold))], r2_file, "fastq")

print("Done")
