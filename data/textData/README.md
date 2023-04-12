## textData

This folder store the papers related to cancer (only keep those paper whose title or keywords contain 'cancer' or 'cancers').


### data summary

Collect 29,590 cancer related papers from 321725 xrxiv paper. (biorxiv: 20397 from 254845, chemrxiv: 479 from 16560, medrxiv: 2918 from 50320, arxiv 5796)
Collect 994,376 cancer related papers from 23899787 pubmed medline baseline paper.
So it's 1,023,966 in total.

### process detail
For xrxiv papers, if it contains 'cancer' or 'cancers' in its ['title','doi','authors','abstract','journal'] field, this paper will be left, else filtered.
For pubmed papers, if it contains 'cancer' or 'cancers' in its ['title','keywords'] field, this paper will be left, else filtered.

### cluster files
```text
.
├── concept2entity.json     # {concept:List[entity:str]}
├── concepts.png            # top 200 most frequent concept words
├── entities.tsv            # text:str  abs_idx(line idx in biobert.jsonl) start_idx(token level idx)   label:str
├── entity2idx.json         # {entity:List[idx:int in entities.tsv]}    (text exact match)
├── final_cluster_res.txt   # cluster res for uniq_entities.txt
└── uniq_entities.txt       # uniq entities (used in clustering)
```

So if you want to use, you should use these files: `concept2entity.json` -> `entity2idx.json` -> `entities.tsv` -> `biobert.jsonl`. Then you can find all the concept positions at token level.