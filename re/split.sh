#!/bin/bash

# desc:	split the dataset into different pieces
# author:	Yangzhe Peng
# date:	2023/04/06


# if you want to change these args outside this bash script, use `export varName='xxx'`

data_filename='../data/textData/ner/biobert.jsonl'
data_name_prefix='../data/textData/ner/splits/biobert.jsonl'
output_name_prefix='../data/textData/re/data.jsonl'

export split_num=384
line_num=$(wc -l $data_filename|cut -d " " -f1)
batch_size=$(python -c "import math;print(math.ceil($line_num/$split_num))")

echo "start spliting"
for((i=0;i<split_num;i++)); do
    s=$((i*batch_size))
    e=$(($((i+1))*batch_size))
    len=$batch_size

    if [ $e -gt "$line_num" ];
    then
        len=$((line_num-s))
    fi
    tail -n +$s $data_filename | head -n "$len" > "$data_name_prefix.split$i"
    printf "\r spliting No.%d piece" "$i"
    
done
wait
echo "done"