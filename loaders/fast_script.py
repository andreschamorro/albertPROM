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
import ngsim
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

_LICENSE = "MIT License"

_HOMEPAGE = ""

_TEMPFILES = []

def _tmp(prefix='gpt_gene.', suffix='.tmp'):
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

class FastConfig(datasets.BuilderConfig):
    """BuilderConfig for NGS."""

    def __init__(
        self,
        reads_names,
        label_classes=None,
        **kwargs,
    ):
        """BuilderConfig for Reads.
        Args:
          data_dir: `string`, the path to the folder containing the fasta files
          **kwargs: keyword arguments forwarded to super.
        """
        super(FastConfig, self).__init__(**kwargs)
        self.reads_names = reads_names
        self.label_classes = label_classes

class FastDataset(datasets.GeneratorBasedBuilder):
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
        FastConfig(name="paired_fast", 
            version=VERSION, 
            description="Paired-End Reads", 
            reads_names=["read_1", "read_2"], 
            label_classes=None),
        FastConfig(name="single_fast", 
            version=VERSION, 
            description="Single-End Reads", 
            reads_names=["read_1"], 
            label_classes=None),
    ]

    def _info(self):
        if self.config.name.startswith("paired"):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "read_1": datasets.Value("string"),
                    "read_2": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif self.config.name.startswith("single"):  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "read_1": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        else:
            features = datasets.Features(
                {
                    "read_1": datasets.Value("string"),
                    "read_2": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
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
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")

        data_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "fastq": data_files,
                    "split": "train"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators` "block_size": [int(bs) for bs in row.blockSizes.split(',')],
    def _generate_examples(self, fastq, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        if self.config.name.startswith('single'):
            with open(fastq['read_1'][0], 'r') as r1_file:
                for i, r1 in enumerate(SeqIO.parse(r1_file, 'fastq')):
                    yield i, {
                            "read_1": r1.seq,
                            }
        if self.config.name.startswith('paired'):
            with open(fastq['read_1'][0], 'r') as r1_file, open(fastq['read_2'][0], 'r') as r2_file:
                for i, (r1, r2) in enumerate(zip(SeqIO.parse(r1_file, 'fastq'), SeqIO.parse(r2_file, 'fastq'))):
                    yield i, {
                            "read_1": r1.seq,
                            "read_2": r2.seq,
                            }
