#!/bin/bash
# desc:	merge re result files into one file
# author:	Yangzhe Peng
# date:	2023/04/12

split_dir="../data/textData/re/splits/"
files=( $(ls -v $split_dir) )

data_filename="../data/textData/re/data.jsonl"

#clear previous output
truncate -s 0 $data_filename

for item in "${files[@]}"; do
    echo "$split_dir$item"
    cat "$split_dir$item" >> $data_filename
done
