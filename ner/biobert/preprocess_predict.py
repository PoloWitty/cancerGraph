# preprocess paper's textData file for MultiHeadNER biobert model
# author: Yangzhe Peng
# date: 2023/03/07

import tqdm
import subprocess
import json
import ipdb
from nltk.tokenize import word_tokenize
import re
import os

def get_line_num(filename):
    out = subprocess.getoutput(f'wc -l {filename}|cut -f1 -d " "')
    return int(out)

if __name__=='__main__':
    paperData_filename = '../../data/textData/merged/data.jsonl'
    output_filename = '../../data/textData/merged/tokenized_data3.jsonl'
    # paperData_filename = '../../data/textData/merged/tokenized_data.jsonl'
    # output_filename = '../../data/textData/merged/tokenized_data2.jsonl'
    open(output_filename,'w').close()
    output_fp = open(output_filename,'a')
    with open(paperData_filename) as fp:
        for _ in tqdm.trange(get_line_num(paperData_filename),desc='processing'):
            line = fp.readline().strip()
            obj = json.loads(line)
            obj['title'] = word_tokenize(obj['title'])
            for word in ['Purpose','Purposes','Objectives','Objective','Background','PURPOSE','BACKGROUND','OBJECTIVES','Abstract','ABSTRACT']:
                if obj['abstract'].startswith(word):
                    obj['abstract'] = obj['abstract'][len(word):]
            obj['abstract'] = word_tokenize(obj['abstract'])
            obj['pubdate'] = obj['pubdate'][:4]
            output_fp.write(json.dumps(obj)+'\n')
    print('done')