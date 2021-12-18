# Model_path=transformer_mimic_model_2
# echo $Model_path
# log_file2=$Model_path.testlog
# result_file=$Model_path.result
# gpus=2
#
#  python train.py \
#  -task abs \
#  -mode test \
#  -batch_size 3000 \
#  -test_batch_size 500 \
#  -bert_data_path ../bert_data2/radiology/radiology \
#  -log_file ../logs/$log_file2 \
#  -model_path $Model_path  \
#  -sep_optim true \
#  -use_interval true \
#  -visible_gpus $gpus \
#  -max_pos 512 \
#  -max_length 200 \
#  -alpha 0.95 \
#  -min_length 6 \
#  -result_path ../logs/$result_file \
#  -test_from transformer_mimic_model_2/model_step_10000.pt


Model_path=transformer_mimic_model_seed_1122
echo $Model_path
log_file2=$Model_path.testlog
result_file=$Model_path.result
gpus=1
python train.py -task abs -mode test -batch_size 1000 -test_batch_size 500 \
-bert_data_path ../bert_data2/radiology/radiology -log_file ../logs/$log_file2 \
-model_path $Model_path -sep_optim true -use_interval true -visible_gpus $gpus \
-max_pos 512 -max_length 200 -alpha 0.95 -min_length 6 -result_path ../logs/$result_file \
-test_from $Model_path/model_step_10000.pt
