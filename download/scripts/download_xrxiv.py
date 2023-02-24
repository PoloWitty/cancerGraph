from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv

if __name__=='__main__':
    path = '../xrxiv/'
    medrxiv(path+'medrxiv.jsonl')  #  Takes ~30min and should result in ~35 MB file
    biorxiv(path+'biorxiv.jsonl')  # Takes ~1h and should result in ~350 MB file
    chemrxiv(path+'chemrxiv.jsonl')  #  Takes ~45min and should result in ~20 MB file