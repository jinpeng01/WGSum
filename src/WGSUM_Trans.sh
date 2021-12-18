
Model_path=transformer_openi_model

# gpus=4,5,6,7
gpus=4
log_file=$Model_path.log
DATA_PATH=../bert_openI/radiology/radiology

python train.py \
-mode train -accum_count 5 \
-batch_size 300 \
-bert_data_path $DATA_PATH \
-dec_dropout 0.1 \
-log_file ../logs/$log_file \
-lr 0.05 \
-model_path $Model_path \
-save_checkpoint_steps 200 \
-seed 777 \
-sep_optim false \
-train_steps 20000 \
-use_bert_emb true \
-use_interval true \
-warmup_steps 8000  \
-visible_gpus $gpus \
-max_pos 512 \
-report_every 50 \
-enc_hidden_size 512  \
-enc_layers 6 \
-enc_ff_size 2048 \
-enc_dropout 0.1 \
-dec_layers 6 \
-dec_hidden_size 512 \
-dec_ff_size 2048 \
-encoder baseline \
-task abs



Model_path=transformer_openi_model
echo $Model_path
log_file2=$Model_path.testlog
result_file=$Model_path.result
gpus=1
python train.py \
-task abs \
-mode validate \
-batch_size 3000 \
-test_batch_size 500 \
-bert_data_path DATA_PATH \
-log_file ../logs/$log_file2 \
-model_path $Model_path \
-sep_optim true \
-use_interval true \
-visible_gpus $gpus \
-max_pos 512 \
-max_length 50 \
-alpha 0.95 \
-min_length 6 \
-result_path ../logs/$result_file \
-test_all

