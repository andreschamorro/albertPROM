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
import pandas as pd
from Bio import SeqIO

import datasets

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_DESCRIPTION = """\
        This Gene annotaded Genome Reference dataset is designed to create a NLP like model.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/grc"

_LICENSE = "MIT License"

_URLS = {
    "default" : {
        "genome": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "annotation": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_40/gencode.v40.annotation.gff3.gz", 
    }
}

_FILES = {
    "default" : {
        "genome": "hg38.fa.gz",
        "annotation": "gencode.v40.annotation.gtf.gz", 
    },
    "transcripts" : {
        "transcripts_bed": "transcripts.bed12",
        "transcripts_seq": "transcripts.fa",
    }
}


def _read_gtf_to_df(_gtf_file) -> pd.DataFrame:
    """Create a pandas dataframe.
    By the pandas library the gff3 file is read and
    a pd dataframe with the given column-names is returned."""
    df = pd.read_csv(
        _gtf_file,
        sep="\t",
        comment="#",
        names=[
            "seq_id",
            "source",
            "type",
            "start",
            "end",
            "score",
            "strand",
            "phase",
            "attributes",
        ],
        dtype = {
            "seq_id": "string",
            "source": "string",
            "type": "string",
            "start": "int",
            "end": "int",
            "score": "string",
            "strand": "string",
            "phase": "string",
            "attributes": "string"
        }
    )
    return df

def _bed12_to_df(_bed12_file) -> pd.DataFrame:
    """Create a pandas dataframe.
    By the pandas library the gff3 file is read and
    a pd dataframe with the given column-names is returned."""
    df = pd.read_csv(
        _bed12_file,
        sep="\t",
        comment="#",
        names=[
            "chrom",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
            "blockSizes",
            "blockStarts"
        ],
        dtype = {
            "chrom": "string",
            "start": "int",
            "end": "int",
            "name": "string",
            "score": "string",
            "strand": "string",
            "thickStart": "int",
            "thickEnd": "int",
            "itemRgb": "object",
            "blockCount": "object",
            "blockSizes": "object",
            "blockStarts": "object"
        }
    )
    return df

def _get_attribute(cell, attrs):
    attrs_dir = {}
    for att in attrs:
        attrs_dir[att] = list(filter(lambda x: att in x, cell.split(";")))[0].strip().split(" ")[1].replace('"', '')
    return attrs_dir


def exons_zip(exs):
    exon_count = len(exs)
    exon_sizes = []
    exon_stars = []
    rel_start = None
    rel_end = None
    exs_sorted = exs.sort_values(by=['exon_number'], ascending=(exs.strand.iloc[0] == '+'))
    for i, row in enumerate(exs_sorted.itertuples()):
        exon_sizes.append(row.end - row.start + 1)
        if i == 0:
            exon_stars.append(0)
            rel_start = row.start - 1
            rel_end = row.end
        else:
            exon_stars.append(row.start - 1 - rel_start)
            rel_end = row.end
    exon_sizes = ",".join([str(i) for i in exon_sizes])
    exon_stars = ",".join([str(i) for i in exon_stars])

    return pd.Series({
        'chrom': exs.seq_id.iloc[0], 'start': rel_start, 'end': rel_end,
        'name': exs.transcript_name.iloc[0], 'score': 0, 'strand': exs.strand.iloc[0],
        'thickStart': rel_start, 'thickEnd': rel_end, 'itemRgb': "255,0,0",
        'blockCount': exon_count, 'blockSizes': exon_sizes, 'blockStarts': exon_stars
    })

def to_bed12(exon_df, num_processes=None):
    # If num_processes is not specified, default to #machine-cores
    if num_processes==None:
        num_processes = cpu_count()

    # 'with' context manager takes care of pool.close() and pool.join() for us
    with Pool(num_processes) as pool:
        groups = [group for name, group in exon_df.groupby(['transcript_name'])]
        # pool.map returns results as a list
        results = pool.map(exons_zip, groups)
    return pd.concat(results, axis=1).T

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

def _extract(in_file, out_file=None):
    if in_file.lower().endswith(('.bgz', '.gz')):
        try:
            from Bio import bgzf
            from Bio import __version__ as bgzf_version
            from packaging.version import Version
            if Version(bgzf_version) < Version('1.73'):
                raise ImportError
        except ImportError:
            raise ImportError(
                    "BioPython >= 1.73 must be installed to read block gzip files.")
        else:
            try:
                # mutable mode is not supported for bzgf anyways
                f_in = bgzf.BgzfReader(in_file, "rb")
            except (ValueError, IOError):
                try:
                    import gzip
                    f_in = gzip.open(in_file, "rb")
                except:
                    raise UnsupportedCompressionFormat(
                        "Compressed is only supported in BGZF or GZIF format. Use "
                        "the samtools bgzip utility (or gzip) to "
                        "compress your files."
                    )
    if out_file is None:
        basename = os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])
        out_file = _tmp(basename[0]+'.', basename[1])

    with open(out_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_file


class AnnotationDataset(datasets.GeneratorBasedBuilder):
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
        datasets.BuilderConfig(name="hg38", version=VERSION, description="This part of my dataset covers a first domain"),
    ]

    DEFAULT_CONFIG_NAME = "default"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _preprocessing(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        data_files = {}
        for k, f in _FILES[self.config.name].items():
            data_files[k] = os.path.join(data_dir, f)
            if not os.path.exists(data_files[k]):
                raise FileNotFoundError(
                    "{data_files[k]} does not exist. Make sure you insert a manual dir that includes the file name {f}. Manual download instructions: {self.manual_download_instructions})"
                )
        for k, f in _FILES["transcripts"].items():
            data_files[k] = os.path.join(data_dir, f)

        if not os.path.exists(data_files['transcripts_bed']) or (os.path.getmtime(
                data_files['transcripts_bed']) < os.path.getmtime(data_files['annotation'])):
            logger.info("Pasing GFF file...")
            exon_df = _read_gtf_to_df(data_files['annotation'])
            exon_df = exon_df.loc[exon_df.type.isin(['exon'])].copy().reset_index(drop=True)
            attributes_list = ['gene_id', 'gene_name', 'gene_type', 'transcript_name', 'exon_id', 'exon_number']
            exon_df[attributes_list] = pd.DataFrame(
                    exon_df.attributes.apply(lambda cell: _get_attribute(cell, attributes_list)).to_list(), 
                    columns = attributes_list, dtype="string")
            exon_df.exon_number = exon_df.exon_number.astype("int")
            exon_df.drop('attributes', inplace=True, axis=1)
            exon_bed12 = to_bed12(exon_df, num_processes=8)
            exon_bed12.drop(exon_bed12[exon_bed12.start > exon_bed12.end].index, inplace=True)
            exon_bed12.to_csv(data_files['transcripts_bed'], sep='\t', header=False, index=False)

        if not os.path.exists(data_files['transcripts_seq']) or (os.path.getmtime(
                data_files['transcripts_seq']) < os.path.getmtime(data_files['transcripts_bed'])):
            logger.info("Extract fasta file for bedtools...")
            extract_genome = _extract(data_files['genome'])

            logger.info("Run bedtools...")
            cmd = ["bedtools", "getfasta",
                    "-fi", extract_genome,
                    "-bed", data_files['transcripts_bed'], 
                    "-fo", data_files['transcripts_seq'], 
                    "-nameOnly", "-s", "-split"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            if p.returncode:
                raise ValueError("bedtools says: %s" % stderr)

        return data_files

    def _info(self):
        if self.config.name == "hg38":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "feature": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "transcript_name": datasets.Value("string"),
                    "block_size": datasets.features.Sequence(datasets.Value("int64")) 
                    # These are the features of your dataset like images, labels ...
                }
            )
        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "feature": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "transcript_name": datasets.Value("string"),
                    "block_size": datasets.features.Sequence(datasets.Value("int64")) 
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
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
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = self._preprocessing(dl_manager)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "transcripts_bed": data_files['transcripts_bed'],
                    "transcripts_seq": data_files['transcripts_seq'],
                    "split": "train"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators` "block_size": [int(bs) for bs in row.blockSizes.split(',')],
    def _generate_examples(self, transcripts_bed, transcripts_seq, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(transcripts_seq) as fasta:
            fasta_dict = {seq.id[:-3]: seq.seq for seq in SeqIO.parse(fasta, "fasta")}
            transcripts_bed = _bed12_to_df(transcripts_bed)
            for key, row in enumerate(transcripts_bed.itertuples()):
                if self.config.name == "default":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "feature": fasta_dict[row.name],
                        "type": "transcriptome",
                        "transcript_name": row.name,
                        "block_size": [int(bs) for bs in row.blockSizes.split(',')],
                    }
                else:
                    yield key, {
                        "feature": fasta_dict[row.name],
                        "type": "transcriptome",
                        "transcript_name": row.name,
                        "block_size": [int(bs) for bs in row.blockSizes.split(',')],
                    }
