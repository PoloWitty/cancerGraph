"""
desc:	get triples from re and cluster result
author:	Yangzhe Peng
date:	2023/04/12
"""

import typer
from pathlib import Path
import subprocess
import tqdm
import json
import pdb

def get_line_num(filename:str):
    out = subprocess.getoutput(f'wc -l {filename}|cut -d " " -f1')
    return int(out)

def main(
    reResult_filename:Path=Path('../data/textData/re/data.jsonl'),
    clusterResult_dir:Path=Path('../data/textData/cluster/'),
    output_filename:Path=Path('../data/graphData/triples.tsv')
):
    '''
        read relation extractoin result and convert it to concept level, output the graph data
        param:
            reResult_filename
            clusterResult_dir
            output_dir: the path to store the output
    '''
    assert reResult_filename.is_file(),f'{reResult_filename} should be a re result file'
    assert clusterResult_dir.is_dir(),f'{clusterResult_dir} should be a cluster result dir'
    
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    log_filename = 'gen_triple.log'
    open(log_filename,'w').close()
    log_fp = open(log_filename,'a')
    
    # read concept2entity file
    print('loading concept2entity...')
    concept2entity = json.load(open(clusterResult_dir/'concept2entity.json'))
    entity2concept = {}
    for k,v in tqdm.tqdm(concept2entity.items(),desc='converting concept2entity to entity2concept'):
        for e in v:
            entity2concept[e] = k
    
    with open(reResult_filename) as fp:
        for _ in tqdm.trange(get_line_num(reResult_filename),desc='processing',dynamic_ncols=True):
            line = fp.readline().strip()
            obj = json.loads(line)
            rels = obj['re']
            text = obj['abstract']
            
            for rel in rels:
                triple,p = rel
                h_pos,r,t_pos = triple
                h = ' '.join(text[h_pos[0]:h_pos[1]+1])
                t = ' '.join(text[t_pos[0]:t_pos[1]+1])
                try:
                    h = entity2concept[h]
                    t = entity2concept[t]
                except KeyError:
                    print(f'{h} or {t} not found in entity2concept',file=log_fp)
                
                output_fp.write(f'{h}\t{r}\t{t}\t{p}\n')
            
if __name__=='__main__':
    typer.run(main)