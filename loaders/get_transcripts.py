import sys
import gzip
import os
import tempfile
from multiprocessing import Pool, cpu_count
import subprocess
import shutil
import six
import argparse
import pandas as pd
from Bio import SeqIO

main_parse = argparse.ArgumentParser()

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
        'name': exs.seq_id.iloc[0] + '_' + exs.transcript_name.iloc[0],
        'score': 0, 'strand': exs.strand.iloc[0], 'thickStart': rel_start,
        'thickEnd': rel_end, 'itemRgb': "255,0,0",
        'blockCount': exon_count, 'blockSizes': exon_sizes, 'blockStarts': exon_stars
    })

def to_bed12(exon_df, num_processes=None):
    # If num_processes is not specified, default to #machine-cores
    if num_processes==None:
        num_processes = cpu_count()

    # 'with' context manager takes care of pool.close() and pool.join() for us
    with Pool(num_processes) as pool:
        groups = [group for name, group in exon_df.groupby(['transcript_name', 'seq_id'])]
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

def main(arg):

    data_files = {}
    data_files['genome'] = os.path.join(arg.dir, arg.ref)
    if not os.path.exists(data_files['genome']):
        raise FileNotFoundError(
            "Genome files does not exist. Make sure you insert a manual dir that includes the file name {f}. Manual download instructions: {self.manual_download_instructions})"
        )
    data_files['annotation'] = os.path.join(arg.dir, arg.ann)
    if not os.path.exists(data_files['annotation']):
        raise FileNotFoundError(
            "Genome files does not exist. Make sure you insert a manual dir that includes the file name {f}. Manual download instructions: {self.manual_download_instructions})"
        )

    data_files['transcripts_bed'] = os.path.join(arg.dir, arg.obe)
    data_files['transcripts_seq'] = os.path.join(arg.dir, arg.ofa)

    if not os.path.exists(data_files['transcripts_bed']) or (os.path.getmtime(
            data_files['transcripts_bed']) < os.path.getmtime(data_files['annotation'])):
        sys.stdout.write('Parssing GFF file..\n')
        sys.stdout.flush()
        exon_df = _read_gtf_to_df(data_files['annotation'])
        exon_df = exon_df.loc[exon_df.type.isin(['exon'])].copy().reset_index(drop=True)
        attributes_list = ['gene_id', 'gene_name', 'gene_type', 'transcript_name', 'exon_id', 'exon_number']
        exon_df[attributes_list] = pd.DataFrame(
                exon_df.attributes.apply(lambda cell: _get_attribute(cell, attributes_list)).to_list(), 
                columns = attributes_list, dtype="string")
        exon_df.exon_number = exon_df.exon_number.astype("int")
        exon_df.drop('attributes', inplace=True, axis=1)
        to_bed12(exon_df, num_processes=arg.threads).to_csv(data_files['transcripts_bed'], sep='\t', header=False, index=False)

    if not os.path.exists(data_files['transcripts_seq']) or (os.path.getmtime(
            data_files['transcripts_seq']) < os.path.getmtime(data_files['transcripts_bed'])):
        sys.stdout.write("Extract fasta file for bedtools...\n")
        sys.stdout.flush()
        extract_genome = _extract(data_files['genome'])

        sys.stdout.write("Run bedtools...\n")
        sys.stdout.flush()
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

main_parse.add_argument('-d', '--dir', type=str, default='.', help='path to datadir')
main_parse.add_argument('-r', '--ref', type=str, default='hg38.fa.gz', help='Reference Genome')
main_parse.add_argument('-a', '--ann', type=str, default='gencode.v40.annotation.gtf.gz', help='Annotation')
main_parse.add_argument('-b', '--obe', type=str, default='transcripts.bed12', help='Annotation')
main_parse.add_argument('-f', '--ofa', type=str, default='transcripts.fa', help='Annotation')
main_parse.add_argument('-t', '--threads', type=int, default=8, help='Threads')
main_parse.set_defaults(func=main)

if __name__ == '__main__':
    args = main_parse.parse_args()
    args.func(args)
