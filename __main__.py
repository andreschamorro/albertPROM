import click
from scripts import run_clm

@click.command("fastdna")
def run_clm():
    return run_clm.main() 
