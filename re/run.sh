# desc:	infer the splited data from start idx to end idx
# author:	Yangzhe Peng
# date:	2023/04/06

# arg1: machine_idx
# arg2: len: process how many files on each machine (note that one process only process one file here)
machine_idx=${1:-0}
len=${2:-8}

# if you want to change these args outside this bash script, use `export varName='xxx'`
data_name_prefix=${data_name_prefix:-'../data/textData/ner/splits/biobert.jsonl'}
output_name_prefix=${output_name_prefix:-'../data/textData/re/data.jsonl'}

echo "machine $machine_idx start predicting"
for((i=$[$machine_idx*$len];i<$[$[$machine_idx+1]*$len];i++)); do
    # --position $[$i-$[$machine_idx*$len]] 
    
    # skip those already processed
    if [ `wc -l "$output_name_prefix.split$i"|cut -d " " -f1` == "2667" ]; then
        continue
    fi
    
    echo "processing No.$i "
    python relation_extraction.py --nerresult-filename "$data_name_prefix.split$i" --output-filename "$output_name_prefix.split$i" 
done
wait
echo "done"

# kill all the subprocess
# pkill -P $$
# ps -ef | grep "python relation_extraction.py"|cut -d " " -f2 | xargs -I '{}' kill {}