##################
# 中文无监督方法
#################
mkdir -p ./logs
mkdir -p ./output
dataset_dir=../datasets/processed

# 中文语义相似数据集benchmark
eval_list=csts_dev#${dataset_dir}/Chinese-STS-B/dev.tsv,csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv


:<<EOF
# SimCSE
exp_name=simclue
task_type=single
loss_type=simcse
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 1 --train-batch-size 64 --eval-batch-size 32 --learning-rate 3e-05 \
    --model-save-dir output/training_simclue-single_simcse \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# TSDAE
# with cls pooling
exp_name=simclue
task_type=single
loss_type=tsdae
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 --pooling cls \
    --num-epochs 2 --train-batch-size 128 --eval-batch-size 32 --learning-rate 3e-05 \
    --model-save-dir output/training_simclue-single_tsdae \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Enhanced-SimCSE
exp_name=simclue
task_type=single
loss_type=esimcse
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 1 --train-batch-size 64 --eval-batch-size 32 --learning-rate 3e-05 \
    --model-save-dir output/training_simclue-single_esimcse \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=single
loss_type=ct
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 1 --train-batch-size 64 --eval-batch-size 32 --learning-rate 3e-05 \
    --model-save-dir output/training_simclue-single_ct \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=single
loss_type=ct2
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 1 --train-batch-size 64 --eval-batch-size 32 --learning-rate 3e-05 \
    --model-save-dir output/training_simclue-single_ct2 \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=single
loss_type=mlm
export CUDA_VISIBLE_DEVICES=0
python src/train_mlm.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.single.tsv \
    --test-type sts --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 2 --train-batch-size 128 --eval-batch-size 32 --learning-rate 2e-05 \
    --mlm-prob 0.15 --do-whole-word-mask \
    --model-save-dir output/training_simclue-single_mlm \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1
EOF

# 中文语义相似数据集benchmark
model_list=csts_base#output/training_csts_benchmark-sts,csts_best#output/training_csts_allnlizh_pkuparaph_pawsx-sts_default,simclue_v1#output/training_simclue-qmc_default,simcse#output/training_simclue-single_simcse,esimcse#output/training_simclue-single_esimcse,tsdae#output/training_simclue-single_tsdae,mlm#output/training_simclue-single_mlm,ct#output/training_simclue-single_ct,ct2#output/training_simclue-single_ct2
log_file=logs/eval.log-zh_unsup.log
nohup python src/eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 64 > ${log_file} 2>&1 &
