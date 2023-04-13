"""
desc:	generate paper info for graphData(add pid to each paper)
author:	Yangzhe Peng
date:	2023/04/13
"""

import subprocess
import typer
from pathlib import Path
import tqdm
import json

def get_line_num(filename):
    out = subprocess.getoutput(f'wc -l {filename}|cut -d " " -f1')
    return int(out)

def main(
    paper_filename:Path = Path('../data/textData/merged/tokenized_data3.jsonl'),
    output_filename:Path = Path('../data/graphData/papers.tsv')
):
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    
    with open(paper_filename) as fp:
        for pid in tqdm.trange(get_line_num(paper_filename),desc='processing',dynamic_ncols=True):
            line = fp.readline()
            output_fp.write(f'{pid}\t{line}')


if __name__=='__main__':
    typer.run(main)