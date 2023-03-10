# biobert finetune

- dataset: collected biobert ner dataset 
    - merge them to a single file and give them types as follows:
```json
{
'NCBI-disease': 'Disease',
'BC5CDR-disease':'Disease',
'BC5CDR-chem': 'Drug/Chem',
'BC4CHEMD': 'Drug/Chem',
'JNLPBA': 'Gene/Protein',
'BC2GM': 'Gene/Protein',
'linnaeus': 'Species',
's800': 'Species'
}
```
- finetune hyper-parameters and result can be found in [multiNERHead result](./result/multiNERHead.md) and [singleNERHead result](./result/singleNERHead.md)
    - In general, multiNERHead result is much better than singleNERHead result.