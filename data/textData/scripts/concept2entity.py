"""
desc:	get the concept2entity map
author:	Yangzhe Peng
date:	2023/03/24
"""

import tqdm
import json
import typer
from pathlib import Path
import subprocess
from collections import defaultdict

def get_line_num(filename:Path):
    out = subprocess.getoutput(f'wc -l {filename}|cut -d " " -f1')
    return int(out)

def main(
        cluster_res_filename: Path = Path('./cluster/final_cluster_res.txt'),
        output_filename: Path = Path('./cluster/concept2entity.json')
    ):
    concept2entity = {}
    with open(cluster_res_filename) as fp:
        for _ in tqdm.trange(get_line_num(cluster_res_filename),desc='reading clusters'):
            line = fp.readline().strip()
            mentions = line.split('|')
            
            # choose the min len mention to be the concept
            min_m = mentions[0]
            for m in mentions:
                if len(min_m)>len(m):
                    min_m = m
            if min_m =='neop':
                min_m = 'tumor'
            elif min_m.endswith(' ca'):
                min_m = min_m.replace('ca','cancer')
            elif min_m.startswith('ca '):
                min_m = min_m.replace('ca','cancer')
            
            if min_m.endswith('s') and min_m[:-1] in concept2entity: # meet plural form and singular form already in the dict
                concept2entity[min_m[:-1]]+=mentions
            elif min_m+'s' in concept2entity: # meed singular form and plural form already in the dict
                concept2entity[min_m] = mentions + concept2entity.pop(min_m+'s')
            else: # normal
                concept2entity[min_m] = mentions
    
    json.dump(concept2entity,open(output_filename,'w'),indent=4)

if __name__=='__main__':
    typer.run(main)