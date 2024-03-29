bash run_uie_finetune.bash \
  -v \
  --device 0 \
  --batch 16 \
  --run-time 3 \
  --warmup_ratio 0.06 \
  --data entity/bio_ner \
  --epoch 30 \
  --spot_noise 0.1 \
  --asoc_noise 0.1 \
  --format spotasoc \
  --map_config config/offset_map/closest_offset_en.yaml \
  --model hf_models/uie-base-en \
  --random_prompt