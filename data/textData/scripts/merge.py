# This file merge the jsonl file from xrxiv and pubmed

import tqdm
import glob
import os
import json
import pdb

if __name__=='__main__':
    pubmed_files = glob.glob('../pubmed/pubmed*.xml.gz.parsed')
    xrxiv_files = glob.glob('../xrxiv/*.jsonl')

    os.makedirs('../merged',exist_ok=True)
    output_file = '../merged/data.jsonl'
    open(output_file,'w').close()
    output_fp = open(output_file,'a')

    with tqdm.tqdm(xrxiv_files,desc='merging xrxiv files') as pbar:
        for file in pbar:
            src = file.split('/')[-1].split('.')[0]

            pbar.set_postfix_str(src)

            with open(file) as fp:
                for line in fp:
                    obj = json.loads(line)
                    output_dict = {
                            'title':obj['title'],
                            'abstract':obj['abstract'],
                            'journal':obj['journal'],
                            'authors':obj['authors'],
                            'pubdate':obj['date'],
                            'doi':obj['doi'],
                            'src':src,
                            'pmid':''
                        }
                    output_fp.write(json.dumps(output_dict)+'\n')

    for file in tqdm.tqdm(pubmed_files,desc='merging pubmed files'):
        with open(file) as fp:
            for line in fp:
                obj = json.loads(line)
                output_dict = {
                            'title':obj['title'],
                            'abstract':obj['abstract'],
                            'journal':obj['journal'],
                            'authors':obj['authors'],
                            'pubdate':obj['pubdate'],
                            'doi':obj['doi'],
                            'src':'pubmed',
                            'pmid':obj['pmid']
                }
                output_fp.write(json.dumps(output_dict)+'\n')
    print('done!')