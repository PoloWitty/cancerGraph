DATA_FILE=../../data/textData/merged/tokenized_data3.jsonl
MAX_LENGTH=512
BATCH_SIZE=32
NUM_EPOCHS=25
SEED=1
MODEL_DIR=./exp/MultiNERHead/job1/
OUTPUT_DIR=./result/

export WANDB_PROJECT='CancerGraph_NER'
export WANDB_LOG_MODEL=True
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=True

python predict.py \
    --model_name_or_path ${MODEL_DIR}\
    --predict_file ${DATA_FILE} \
    --text_column_name abstract \
    --preprocessing_num_workers 8 \
    --max_seq_length ${MAX_LENGTH} \
    --pad_to_max_length True \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --save_strategy epoch \
    --seed ${SEED} \
    --data_seed ${SEED} \
    --use_multiNERHead True \
    --do_predict \
    --decode_worker_num 20 
    # --max_predict_samples 512 \
    # --overwrite_output_dir \