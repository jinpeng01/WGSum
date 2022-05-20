Model_path=../models/openi-rerun
echo $Model_path
log_file2=$Model_path/testlog
result_file=$Model_path/result
DATA_PATH=../bert_openI/radiology/radiology


# gpus=0
# python train.py \
# -task abs \
# -mode validate \
# -batch_size 3000 \
# -test_batch_size 500 \
# -bert_data_path $DATA_PATH \
# -log_file ../logs/$log_file2 \
# -model_path $Model_path \
# -sep_optim true \
# -use_interval true \
# -visible_gpus $gpus \
# -max_pos 512 \
# -max_length 50 \
# -alpha 0.95 \
# -min_length 6 \
# -result_path ../logs/$result_file \
# -test_all


gpus=0
python train.py \
-task abs \
-mode test \
-batch_size 3000 \
-test_batch_size 500 \
-bert_data_path $DATA_PATH \
-log_file $log_file2 \
-model_path $Model_path \
-sep_optim true \
-use_interval true \
-visible_gpus $gpus \
-max_pos 512 \
-max_length 50 \
-alpha 0.95 \
-min_length 6 \
-result_path $result_file \
-test_from $Model_path/openi.pt \
