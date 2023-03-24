"""
desc:	draw a word cloud using concept frequency
author:	Yangzhe Peng
date:	2023/03/24
"""

from wordcloud import WordCloud
import typer
from pathlib import Path
import json
from collections import defaultdict
import tqdm
import pdb

def main(
        entity2idx_filename: Path = Path('./cluster/entity2idx.json'),
        concept2entity_filename: Path = Path('./cluster/concept2entity.json'),
        output_path: Path = Path('./cluster')
):
    print('loading entity2idx...')
    entity2idx = json.load(open(entity2idx_filename))
    print('loading concept2entity...')
    concept2entity = json.load(open(concept2entity_filename))

    concept2freq = defaultdict(int)
    errorEntity = []
    for concept,entities in tqdm.tqdm(concept2entity.items(),desc='gathering statistics'):
        for entity in entities:
            try:
                idx = entity2idx[entity]
                concept2freq[concept] += len(idx)
            except:
                errorEntity.append(entity)
    
    print(f'{len(errorEntity)} entities missed')
    print(errorEntity)    
    wc = WordCloud(background_color='white',width=900,height=500, max_words=200,relative_scaling=1)
    wc.generate_from_frequencies(frequencies = concept2freq).to_file(output_path/'concepts.png')
    

if __name__=='__main__':
    typer.run(main)