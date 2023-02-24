## textData

This folder store the papers related to cancer (only keep those paper whose title or keywords contain 'cancer' or 'cancers').


### data summary

Collect 29,590 cancer related papers from 321725 xrxiv paper. (biorxiv: 20397 from 254845, chemrxiv: 479 from 16560, medrxiv: 2918 from 50320, arxiv 5796)
Collect 994,376 cancer related papers from 23899787 pubmed medline baseline paper.
So it's 1,023,966 in total.

### process detail
For xrxiv papers, if it contains 'cancer' or 'cancers' in its ['title','doi','authors','abstract','journal'] field, this paper will be left, else filtered.
For pubmed papers, if it contains 'cancer' or 'cancers' in its ['title','keywords'] field, this paper will be left, else filtered.

