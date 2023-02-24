import paperscraper as ps
import tqdm

if __name__=='__main__':
    cancer = ['cancer','cancers']
    query = [cancer]

    print(ps.QUERY_FN_DICT.keys())

    for i in tqdm.tqdm(ps.QUERY_FN_DICT.keys()):
        try:
            if i == 'pubmed':
                continue
            ps.QUERY_FN_DICT[i](query,output_filepath='../xrxiv/%s.jsonl'%i)
        except:
            print('error when querying %s'%i)