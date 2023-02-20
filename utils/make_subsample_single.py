#!/usr/bin/env python
"""
usage:
    $  ./make_subsample_paired.py PRESENCE_PREFIX ABSENCE_PREFIX SAMPLE_DIR N_SAMPLE RATIO VALSIZE 
"""
import sys, os, random
import subprocess
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

def _processing(s, label):
    s.id = s.id.split('|')[0] + s.id.split('-')[-1]
    s.description = label
    s.name=""
    return s

pop_if = lambda l,i: l.pop() != None if (l and l[-1] == i) else False

all_features = []
all_targets = []
# Presence sample
print("presence processing ...")
p_features = []
pr_count = 0
with open(f"{p_prefix}.fq", "r") as pr_file:
    p_features = [_processing(r, "presence") for r in SeqIO.parse(pr_file, "fastq")]

pr_count = len(p_features)
print(f"presence count: {pr_count}")

n_sample = n_sample if (n_sample < pr_count and n_sample > 0) else pr_count
r_index = random.sample(range(pr_count), n_sample)

all_features = [p_features[i] for i in r_index]
all_targets = [0 for i in r_index]
print("absence processing ...")

n_sample = n_sample * ratio if n_sample > 0 else pr_count * ratio
ab_count = int(subprocess.check_output(f"echo $(cat {a_prefix}.fq|wc -l)/4|bc", shell=True).split()[0])
print(f"absence count: {ab_count}")

r_index = random.sample(range(ab_count), n_sample if n_sample < ab_count else ab_count)
r_index.sort(reverse=True)
all_targets.extend([1 for i in r_index])
with open(f"{a_prefix}.fq", "r") as ar_file:
    all_features.extend([_processing(r, "absence") for i, r in enumerate(SeqIO.parse(ar_file, "fastq")) if pop_if(r_index, i)])
all_targets.extend([1 for i in r_index])

print("All sample processed")
# suffle data
print("Write sample ...")

train_fold, train_y, test_fold, test_y = train_test_split(all_features, all_targets, random_state=1, shuffle=True, stratify=all_targets)
test_fold, _, val_fold, _ = train_test_split(test_fold, test_y, test_size=val_size, random_state=1, stratify=test_y)
# Train sample
try:
   os.makedirs(d_prefix)
except FileExistsError:
   # directory already exists
   pass
with open(os.path.join(d_prefix, f"train_single.fq"), "w") as r_file:
    SeqIO.write(train_fold, r_file, "fastq")

with open(os.path.join(d_prefix, f"test_single.fq"), "w") as r_file:
    SeqIO.write(test_fold, r_file, "fastq")

with open(os.path.join(d_prefix, f"val_single.fq"), "w") as r_file:
    SeqIO.write(val_fold, r_file, "fastq")

print("Done")
