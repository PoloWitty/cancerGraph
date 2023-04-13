"""
desc:	drop those duplicated triples
author:	Yangzhe Peng
date:	2023/04/13
"""

import tqdm
from pathlib import Path
from collections import defaultdict
import typer

from get_triples import get_line_num

def main(
    triples_filename:Path = Path('../data/graphData/triples.tsv'),
    output_filename:Path = Path('../data/graphData/triples_no_duplicate.tsv')
):
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    
    triples = defaultdict(list)
    with open(triples_filename) as fp:
        for i in tqdm.trange(get_line_num(triples_filename),desc='reading triples'):
            line = fp.readline().strip()
            h,r,t,p = line.split('\t')
            triples[f'{h}\t{r}\t{t}'].append(float(p))
    
    
    for k,v in tqdm.tqdm(triples.items(),desc='processing',dynamic_ncols=True):
        p = sum(v)/len(v)
        output_fp.write(f'{k}\t{p}\n')
        

if __name__=='__main__':
    typer.run(main)