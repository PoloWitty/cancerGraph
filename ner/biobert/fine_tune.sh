DATA_DIR=../dataset/processed/multiNERHead/
MAX_LENGTH=192
BATCH_SIZE=32
NUM_EPOCHS=25
SEED=1
SAVE_DIR=./exp/MultiNERHead/
RUN_NAME='MultiNERHead'

export WANDB_PROJECT='CancerGraph_NER'
export WANDB_LOG_MODEL=True
# export WANDB_DISABLED=True

python run_ner.py \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.2 \
    --train_file ${DATA_DIR}/train.json \
    --validation_file ${DATA_DIR}/dev.json \
    --test_file ${DATA_DIR}/test.json \
    --label_column_name ner_tags \
    --text_column_name tokens \
    --preprocessing_num_workers 6 \
    --max_seq_length ${MAX_LENGTH} \
    --pad_to_max_length True \
    --output_dir ${SAVE_DIR}/job1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_strategy epoch \
    --load_best_model_at_end \
    --save_total_limit 5 \
    --seed ${SEED} \
    --data_seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --return_entity_level_metrics \
    --run_name  ${RUN_NAME}\
    --report_to wandb \
    --use_multiNERHead True
    # --max_train_samples 30 \
    # --max_eval_samples 20 \
    # --max_predict_samples 20 \
    # --overwrite_output_dir \