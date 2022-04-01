mkdir -p ./logs
mkdir -p ./output
dataset_dir=../datasets/processed

###############
# 英文无监督表示学习
###############
# 1. 无监督-simcse
exp_name=en_unsup
task_type=single
loss_type=simcse
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_en_unsup-single-simcse \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2. 无监督-esimcse
exp_name=en_unsup
task_type=single
loss_type=esimcse
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_en_unsup-single-esimcse-v2 \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 3. 无监督-tsdae
# pooling with cls
exp_name=en_unsup
task_type=single
loss_type=tsdae
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 --pooling cls \
    --num-epochs 2 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_en_unsup-single-tsdae \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 4. 无监督-ct
exp_name=en_unsup
task_type=single
loss_type=ct
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_en_unsup-single-ct \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 5. 无监督-ct2
exp_name=en_unsup
task_type=single
loss_type=ct2
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_en_unsup-single-ct2 \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 6. 无监督-mlm
exp_name=en_unsup
task_type=single
loss_type=mlm
export CUDA_VISIBLE_DEVICES=0
python src/train_mlm.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sentences_en/wiki1m_for_simcse.txt \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --mlm-prob 0.15 --do-whole-word-mask \
    --model-save-dir output/training_en_unsup-single_mlm \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Evaluation
:<<EOF
# EN baseline
python src/eval.py --eval-mode sts --eval-list sts_test#${dataset_dir}/sts_benchmark/test.tsv \
    --model-list bert-mean#bert-base-uncased,glove_v1#sentence-transformers/average_word_embeddings_glove.6B.300d,glove_v2#sentence-transformers/average_word_embeddings_komninos,w2v#sentence-transformers/average_word_embeddings_levy_dependency,glove_v3#sentence-transformers/average_word_embeddings_glove.840B.300d \
    > logs/eval.en_unsup_baseline.log 2>&1
EOF

python src/eval.py --eval-mode sts --eval-list sts_test#${dataset_dir}/sts_benchmark/test.tsv \
    --model-list simcse#output/training_en_unsup-single-simcse,esimcse#output/training_en_unsup-single-esimcse,tsdae#output/training_en_unsup-single-tsdae,ct#output/training_en_unsup-single-ct,ct2#output/training_en_unsup-single-ct2,mlm#output/training_en_unsup-single_mlm,esimcse_v2#output/training_en_unsup-single-esimcse-v2 \
    --device cuda:0 --batch-size 32 \
    > logs/eval.log-en_unsup.log 2>&1
