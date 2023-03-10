# merge files in src dir and map them to corresponding entity type
# author: Yangzhe Peng
# date: 2023/03/02

import glob
import subprocess
import tqdm
import ipdb
import os
import json

def get_line_num(file_name):
    out = subprocess.getoutput('wc -l %s | cut -f1 -d " "'%file_name)
    return int(out)


if __name__=='__main__':
    dataset2type = {
        'NCBI-disease': 'Disease',
        'BC5CDR-disease':'Disease',
        'BC5CDR-chem': 'Drug/Chem',
        'BC4CHEMD': 'Drug/Chem',
        'JNLPBA': 'Gene/Protein',
        'BC2GM': 'Gene/Protein',
        'linnaeus': 'Species',
        's800': 'Species'
    }
    output_dir = '../processed/NER/'
    os.makedirs(output_dir,exist_ok=True)
    data_dirs = glob.glob('../src/*')
    datasets = []
    for data_dir in data_dirs:
        train_file = data_dir +'/train.tsv'
        dev_file = data_dir +'/devel.tsv'
        test_file = data_dir +'/test.tsv'
        splits = []
        for file in [train_file,dev_file,test_file]:
            with open(file) as fp:
                examples = []
                example = {'tokens':[],'ner_tags':[]}
                for _ in tqdm.trange(get_line_num(file),desc=f'reading {file} '):
                    line = fp.readline().strip()

                    if line == '':
                        if example.get('tokens','')!='':
                            # save example
                            examples.append(example)
                        # new example
                        example = {'tokens':[],'ner_tags':[]}
                        continue

                    word,tag = line.split('\t')

                    example['tokens'].append(word)

                    dataset = file.split('/')[-2]
                    if tag != 'O':
                        tag = tag + '-' + dataset2type[dataset] # B/I/O-tag
                        # tag = tag # B/I/O
                    example['ner_tags'].append(tag)
                    example['dataset_type'] = dataset2type[dataset]
            assert len(examples)!=0 , ipdb.set_trace()
            splits.append(examples)
        datasets.append(splits)

    for i,split in enumerate(['train','dev','test']):
        output_file = output_dir+split+'.json'
        open(output_file,'w').close() # clear previous output
        for j,dataset in enumerate(datasets):
            with open(output_file,'a') as fp:
                for example in tqdm.tqdm(dataset[i],desc=f'writing No. {j} {split} split to {output_file}'):
                    fp.write(json.dumps(example)+'\n')

