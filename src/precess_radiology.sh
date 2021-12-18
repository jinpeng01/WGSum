CUDA_VISIBLE_DEVICES=1 python preprocess.py \
-mode format_to_bert \
-raw_path JSON_PATH \
-save_path ../bert_data/radiology/  \
-lower \
-n_cpus 1 \
-log_file ../logs/preprocess.log \
-type edges_word

