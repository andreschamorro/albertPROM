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
# Presence sample
print("presence processing ...")
p_features = []
pr_count = 0
with open(f"{p_prefix}1.fq", "r") as pr1_file, open(f"{p_prefix}2.fq", "r") as pr2_file:
    p_features = [(_processing(r1, "presence"), _processing(r2, "presence")) 
                  for r1, r2 in zip(SeqIO.parse(pr1_file, "fastq"), SeqIO.parse(pr2_file, "fastq"))]

pr_count = len(p_features)
print(f"presence count: {pr_count}")

n_sample = n_sample if (n_sample < pr_count and n_sample > 0) else pr_count
r_index = random.sample(range(pr_count), n_sample)

all_features = [p_features[i] for i in r_index]
all_targets = [0 for i in r_index]

print("absence processing ...")

n_sample = n_sample * ratio if n_sample > 0 else pr_count * ratio
ab_count = int(subprocess.check_output(f"echo $(cat {a_prefix}1.fq|wc -l)/4|bc", shell=True).split()[0])
print(f"absence count: {ab_count}")

r_index = random.sample(range(ab_count), n_sample if n_sample < ab_count else ab_count)
r_index.sort(reverse=True)
all_targets.extend([1 for i in r_index])

with open(f"{a_prefix}1.fq", "r") as ar1_file, open(f"{a_prefix}2.fq", "r") as ar2_file:
    all_features.extend([(_processing(r1, "absence"), _processing(r2, "absence"))
                         for i, (r1, r2) in enumerate(zip(SeqIO.parse(ar1_file, "fastq"), SeqIO.parse(ar2_file, "fastq"))) 
                            if pop_if(r_index, i)])

print("All sample processed")
# suffle data
print("Write sample ...")

train_fold, test_fold, train_y, test_y = train_test_split(all_features, all_targets, random_state=1, shuffle=True, stratify=all_targets)
test_fold, val_fold, _, _ = train_test_split(test_fold, test_y, test_size=val_size, random_state=1, stratify=test_y)
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
