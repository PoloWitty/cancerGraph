import pubmed_parser as pp
import glob
import os
import concurrent.futures
from tqdm import tqdm
import pdb
import json

def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.
    
    Does not support timeout or chunksize as executor.submit is used internally
    
    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()

def parse_and_process_file(filename):
    """
    This function contains our parsing code. Usually, you would only modify this function.
    """
    # Don't parse authors and references, since we don't need it
    dat = pp.parse_medline_xml(filename, author_list=False, reference_list=False)

    paper_cnt=0
    query_paper_cnt = 0
    query1 = 'cancer'; query2 = 'cancers'
    save_name = '../pubmed/'+filename.split('/')[-1]+'.parsed'
    open(save_name,'w').close()
    output_fp = open(save_name,'a')
    for entry in dat:
        title = entry['title']
        keywords = entry['keywords']
        if entry['abstract']=='':
            continue
        if query1 in title or query1 in keywords or query2 in title or query2 in keywords:
            output_dict = {
                'pmid':entry['pmid'],
                'doi':entry['doi'],
                'title':title,
                'abstract':entry['abstract'],
                'journal':entry['journal'],
                'authors':entry['authors'],
                'pubdate':entry['pubdate']
            }
            output_fp.write(json.dumps(output_dict)+'\n')
            query_paper_cnt += 1
        paper_cnt+=1
        
    return paper_cnt,query_paper_cnt


if __name__ == "__main__":
    # Find all pubmed files in the download dir
    all_filenames = glob.glob("../../../download/pubmed/pubmed*.xml.gz")

    executor = concurrent.futures.ProcessPoolExecutor(os.cpu_count())

    total_paper_cnt = 0
    total_query_paper_cnt = 0

    for paper_cnt,query_paper_cnt in tqdm_parallel_map(executor, parse_and_process_file, all_filenames):
        # NOTE: If you print() here, this might interfere with the progress bar,
        total_paper_cnt += paper_cnt
        total_query_paper_cnt += query_paper_cnt

    with open('./summary.log','w') as fp:
        fp.write('pubmed: \n total paper num: %i\ntotal query paper cnt: %i'%(total_paper_cnt,total_query_paper_cnt))
    print('done')