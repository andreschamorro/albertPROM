import os
import snakemake

def run_salmon(**kwargs):
    import snakemake
    snakefile = os.path.join(os.path.dirname(__file__), "snakemake/snakefile.paired" if kwargs["paired"] else "snakemake/snakefile.single")

    snakemake.snakemake(
        snakefile=snakefile,
        config={
            "input_path": kwargs["inpath"],
            "output_path": kwargs["--outpath"],
            "index": kwargs["--reference"],
            "salmon": os.path.join(os.path.expanduser('~'),".local/bin/salmon"),
            "num_threads" : kwargs["--num_threads"],
            "exprtype": kwargs["--exprtype"],
        },
        quiet=True,
        lock=False
    )
    #with open(os.path.join(kwargs["--outpath"], "expr.csv" ), "r") as inp:
    #    sample_ids = inp.readline().strip().split(',')[1:]
    #with open(os.path.join(kwargs["--outpath"], "condition.csv" ), "w") as oup:
    #    oup.write("SampleID,condition\n")
    #    oup.write(
    #        "\n".join([s+","+"NA" for s in sample_ids]) + "\n"
    #    )
