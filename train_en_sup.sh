###############
# 英文表示学习
###############
mkdir -p ./logs
mkdir -p ./output
dataset_dir=../datasets/processed

:<<EOF
# 1.1 直接在 sts 上训练
exp_name=sts_benchmark
task_type=sts
loss_type=default
nohup python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_sts_benchmark-sts \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 1.2 直接在 sts 上训练基于CircleLoss
exp_name=sts_benchmark
task_type=sts
loss_type=circle
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_sts_benchmark-sts-circle \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.1 直接在 xnli和mnli上训练（分类目标）
exp_name=allnli
task_type=nli
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_allnli-nli \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.2 直接在 xnli和mnli上训练（排序目标）
exp_name=allnli_ranking
task_type=nli
loss_type=ranking
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_allnli_ranking-nli_ranking \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.3 直接在 xnli和mnli上训练（circle目标）
exp_name=allnli
task_type=nli
loss_type=circle
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_allnli-nli-circle \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 3.1 基于 nli 训练模型在 sts 上继续训练
exp_name=sts_benchmark_nli
task_type=sts
loss_type=default
model_path=./archives/training_allnli-nli/2022-02-22_14-24-23
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_sts_benchmark_nli-sts \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 3.2 基于 nli-ranking 训练模型在 sts 上继续训练
exp_name=sts_benchmark_nli_ranking
task_type=sts
loss_type=default
model_path=output/training_allnli_ranking-nli_ranking
nohup python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_sts_benchmark_nli_ranking-sts_default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 3.3 基于 nli-circle 训练模型在 sts 上继续训练
exp_name=sts_benchmark_nli
task_type=sts
loss_type=circle
model_path=output/training_allnli-nli-circle
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training_sts_benchmark_nli_circle-sts-circle \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1
EOF

# Evaluation
python src/eval.py --eval-mode sts --eval-list sts_test#${dataset_dir}/sts_benchmark/test.tsv \
    --model-list sts#output/training_sts_benchmark-sts,sts_v2#output/training_sts_benchmark-sts-circle,allnli#output/training_allnli-nli,allnli_v2#output/training_allnli_ranking-nli_ranking,allnli_v3#output/training_allnli-nli-circle,sts_allnli#output/training_sts_benchmark_nli-sts,sts_allnli_v2#output/training_sts_benchmark_nli_ranking-sts_default,sts_allnli_v3#output/training_sts_benchmark_nli_circle-sts-circle,mpnet#all-mpnet-base-v2 \
    > logs/eval.log-en_sup.log 2>&1
