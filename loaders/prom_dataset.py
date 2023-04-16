# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Reference Genome dataset generator"""


import csv
import json
import gzip
import os
import tempfile
from multiprocessing import Pool, cpu_count
import subprocess
import shutil
import six
import itertools
import random
import numpy as np
from Bio import bgzf, SeqIO

import datasets

logger = datasets.logging.get_logger(__name__)


_READS_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_READS_DESCRIPTION = """\
        This Gene annotaded Genome Reference labeled ngs dataset is designed to create a NLP like model.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/grc"

_LICENSE = "MIT License"

_URLS = {
}

_FILES = {
        "sliding_train" : "transcript.sliding.fa",
        "tata_train" : "hs_pos_TATA.fa",
        "nontata_train" : "hs_pos_nonTATA.fa",
}

_LABELS = {
    "labels": ["promoter", "nonpromoter"],
}

_TEMPFILES = []

def _tmp(prefix='albert_prom.', suffix='.tmp'):
    """
    Makes a tempfile and registers it in the BedTool.TEMPFILES class
    variable.  Adds a "gpt_gene." prefix and ".tmp" extension for easy
    deletion if you forget to call cleanup().
    """
    tmpfn = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=suffix,
        delete=False,
    )
    tmpfn = tmpfn.name
    _TEMPFILES.append(tmpfn)
    return tmpfn

class ReadsConfig(datasets.BuilderConfig):
    """BuilderConfig for NGS."""

    def __init__(
        self,
        seq_names,
        label_column,
        label_classes=None,
        process_label=lambda x: x,
        num_seq=None,
        **kwargs,
    ):
        """BuilderConfig for Reads.
        Args:
          data_dir: `string`, the path to the folder containing the fasta files
          **kwargs: keyword arguments forwarded to super.
        """
        super(ReadsConfig, self).__init__(**kwargs)
        self.seq_names = seq_names 
        self.label_column = label_column
        self.label_classes = label_classes
        self.process_label = process_label
        self.num_seq = num_seq if num_seq else 0

class ReadsDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        ReadsConfig(name="tata_prom", 
            version=VERSION, 
            description="Random in blocks", 
            seq_names=["sequence"], 
            num_seq=None),
        ReadsConfig(name="tata_prom", 
            version=VERSION, 
            description="Random in blocks", 
            seq_names=["sequence"], 
            label_classes=_LABELS["labels"],
            label_column="label",
            num_seq=None),
        ReadsConfig(name="nontata_prom", 
            version=VERSION, 
            description="Random in blocks", 
            seq_names=["sequence"], 
            label_classes=_LABELS["labels"],
            label_column="label",
            num_seq=None),
    ]

    def _preprocessing(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if self.config.name.startswith('sliding'):
            fasta_train = os.path.join(data_dir, _FILES['tata_train'])
            if not os.path.exists(fasta_train):
                raise FileNotFoundError(
                    f"{fasta_train} does not exist. Make sure you insert a manual dir that includes the file name {file}. Manual download instructions: {self.manual_download_instructions})"
                )
        elif self.config.name.startswith('tata'):
            fasta_train = os.path.join(data_dir, _FILES['tata_train'])
            if not os.path.exists(fasta_train):
                raise FileNotFoundError(
                    f"{fasta_train} does not exist. Make sure you insert a manual dir that includes the file name {file}. Manual download instructions: {self.manual_download_instructions})"
                )
        else:
            fasta_train = os.path.join(data_dir, _FILES['nontata_train'])
            if not os.path.exists(fasta_train):
                raise FileNotFoundError(
                    f"{fasta_train} does not exist. Make sure you insert a manual dir that includes the file name {file}. Manual download instructions: {self.manual_download_instructions})"
                )

        return fasta_train

    def _info(self):
        if self.config.name.startswith("sliding"):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "sequence": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif self.config.name.startswith("tata"):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "sequence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
            if self.config.label_classes:
                features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
            else:
                features["label"] = datasets.Value("float32")
        else: 
            # self.config.name.startswith("nontata"):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "sequence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
            if self.config.label_classes:
                features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
            else:
                features["label"] = datasets.Value("float32")
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_READS_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_READS_CITATION,
        )

    def _split_generators(self, dl_manager):
        fasta_train = self._preprocessing(dl_manager)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "fasta": fasta_train,
                    "split": "train"
                },
            ),
        ]

    def _neggen(self, seq, num_part=20, keep=8,  prob=1.0):
        if random.random() < (1.0-prob):
            return seq
        length = len(seq)
        # get part
        part_len = length // num_part
        if part_len * num_part < length:
            num_part += 1

        iterator = np.arange(num_part)
        keep_parts = random.sample(list(iterator), k=keep)

        outpro = list()
        for it in iterator:
            start = it * part_len
            pro_part = seq[start:start + part_len]
            if it in keep_parts:
                outpro.extend(pro_part)
            else:
                pro_part = random.choices(['A', 'C', 'G', 'T'], k=len(pro_part))
                outpro.extend(pro_part)
        return "".join(outpro)

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators` "block_size": [int(bs) for bs in row.blockSizes.split(',')],
    def _generate_examples(self, fasta, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        if self.config.name.startswith('sliding'):
            with open(fasta, 'r') as fa_file:
                for i, seq in enumerate(SeqIO.parse(fa_file, 'fasta')):
                    yield i, {"sequence": seq.seq,}
            with open(fasta, 'r') as fa_file:
                for i, seq in enumerate(SeqIO.parse(fa_file, 'fasta')):
                    yield 2*i, {
                            "sequence": seq.seq,
                            "label": "promoter",
                            }
                    yield 2*i + 1, {
                            "sequence": self._neggen(seq.seq),
                            "label": "nonpromoter",
                            }
