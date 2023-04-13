"""
desc:	generate concept file from cluster result
author:	Yangzhe Peng
date:	2023/04/12
"""

import typer
import json
from pathlib import Path
from collections import defaultdict,Counter
import tqdm
import pdb

def main(
    clusterResult_dir:Path=Path('../data/textData/cluster/'),
    output_filename:Path=Path('../data/graphData/concepts.tsv')
):
    '''
        read named entity extraction result and convert it to concept level, output the concepts and their properties.
        This script will determine the type of concept and get concept2pid map.
        param:
            nerResult_filename
            clusterResult_dir
            output_dir: the path to store the output
    '''
    assert clusterResult_dir.is_dir(),f'{clusterResult_dir} should be a cluster result dir'
    
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    open('missed.log','w').close()
    log_fp = open('missed.log','a')
    
    # read entities.tsv
    with open(clusterResult_dir/'entities.tsv') as fp:
        print('loading entities...')
        entities = fp.readlines()
    
    
    # determine each entity type
    print('loading entity2idx...')
    entity2idx = json.load(open(clusterResult_dir/'entity2idx.json'))
    
    entity2pid = defaultdict(set)
    entity2type = dict()
    for entity,indices in tqdm.tqdm(entity2idx.items(),desc='getting entity2pid and entity2type'):
        types = []
        for idx in indices:
            line = entities[idx-1].strip() # idx in entity2idx start from 1
            _,pid,s_pos,t = line.split('\t')
            entity2pid[entity].add(pid)
            types.append(t)
        
        # vote for the type of this entity
        types = Counter(types)
        entity2type[entity] = types.most_common(1)[0][0]
        
        
    # determine each concept type
    print('loading concept2entity...')
    concept2entity = json.load(open(clusterResult_dir/'concept2entity.json'))
    
    for concept,entities in tqdm.tqdm(concept2entity.items(),desc='getting concept2type and concept2pid & write to output'):
        types = []
        entities2pid = {}
        for entity in entities:
            entities2pid[entity] = list(entity2pid[entity])
            try:
                types.append(entity2type[entity.strip()])
            except KeyError:
                log_fp.write(f'{entity} not found in entity2type\n')
                
        if len(types)==0:
            continue
        # vote for the type of this concept
        types = Counter(types)
        t = types.most_common(1)[0][0]
        output_fp.write(f'{concept}\t{t}\t{json.dumps(entities2pid)}\n')
        

if __name__=='__main__':
    typer.run(main)