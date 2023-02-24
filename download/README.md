## pubmed
This is actually pubmed medline baseline data, which can be downloaded from https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/. It's recommand to download them using FileZilla instead of `wget`.

Note: The downloaded file should be checked using md5sum code first. See more in [checksum.py](./scripts/checksum.py)

This should take ~5h and result in ~38GB files which contain 23899787 papers in total.

## xrxiv
This folder include biorxiv, chemrxiv and medrxiv. This can be downloaded using paper_scraper. See more in [download_xrxiv.py](./scripts/download_xrxiv.py)

This should take ~2.5h and result in 589MB files which contain 321725 papers in total (up to 2023/02/21).