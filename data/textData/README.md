## textData

This folder store the papers related to cancer (only keep those paper whose title or keywords contain 'cancer' or 'cancers').


### data summary

Collect 29,590 cancer related papers from 321725 xrxiv paper.
Collect 994,376 cancer related papers from 23899787 pubmed medline baseline paper.
So it's 1,023,966

### process detail
For xrxiv papers, if it contains 'cancer' or 'cancers' in its ['title','doi','authors','abstract','journal'] field, this paper will be left, else filtered.

For pubmed papers, if it contains 'cancer' or 'cancers' in its ['title','keywords'] field, this paper will be left, else filtered.

