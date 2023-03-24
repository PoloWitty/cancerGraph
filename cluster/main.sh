
python utils/generate_use_data.py --ner_path ../data/textData/cluster/uniq_entities.txt --save_dir ./use_data/
python utils/generate_faiss_index.py --model_name_or_path GanjinZero/coder_eng_pp --save_dir ./use_data/ --topk 20
python utils/clustering.py --use_data_dir ./use_data/ --result_dir ./result/ --similarity_threshold 0.9
python utils/ratio_cut.py