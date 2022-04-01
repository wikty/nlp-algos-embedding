###############
# 中文表示学习
# - 各个任务监督学习
###############
mkdir -p ./logs
mkdir -p ./output
dataset_dir=../datasets/processed

exp_name=csts_benchmark
task_type=sts
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-csts_benchmark-sts_default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=ocnli
task_type=nli
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/ocnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-ocnli-nli-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=ocnli_rank
task_type=nli
loss_type=ranking
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/ocnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-ocnli_rank-nli-ranking \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=cmnli
task_type=nli
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/cmnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-cmnli-nli-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=cmnli_rank
task_type=nli
loss_type=ranking
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/cmnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-cmnli_rank-nli-ranking \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=csnli
task_type=nli
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/csnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-csnli-nli-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=csnli_rank
task_type=nli
loss_type=ranking
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/csnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-csnli_rank-nli-ranking \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=pkuparaph
task_type=qmc
loss_type=ranking
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/PKU-Paraphrase-Bank/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-pkuparaph-qmc-ranking \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=afqmc
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/afqmc_public/train.tsv \
    --dev-file ${dataset_dir}/afqmc_public/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-afqmc-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=lcqmc
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/lcqmc/train.tsv \
    --dev-file ${dataset_dir}/lcqmc/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-lcqmc-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=bqcorpus
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/bq_corpus/train.tsv \
    --dev-file ${dataset_dir}/bq_corpus/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-bqcorpus-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=pawsx
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/paws-x-zh/train.tsv \
    --dev-file ${dataset_dir}/paws-x-zh/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-pawsx-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=xiaobu
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/oppo-xiaobu/train.tsv \
    --dev-file ${dataset_dir}/oppo-xiaobu/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-xiaobu-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=qbqtc
task_type=qmc
loss_type=default
python src/train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/QBQTC/train.tsv \
    --dev-file ${dataset_dir}/QBQTC/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir output/training-qbqtc-qmc-default \
    > logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Eval 各任务之间zero-shot能力
model_list=csts#output/training-csts_benchmark-sts_default,ocnli#output/training-ocnli-nli-default,ocnli_rank#output/training-ocnli_rank-nli-ranking,cmnli#output/training-cmnli-nli-default,cmnli_rank#output/training-cmnli_rank-nli-ranking,csnli#output/training-csnli-nli-default,csnli_rank#output/training-csnli_rank-nli-ranking,pku#output/training-pkuparaph-qmc-ranking,afqmc#output/training-afqmc-qmc-default,lcqmc#output/training-lcqmc-qmc-default,bqcorpus#output/training-bqcorpus-qmc-default,pawsx#output/training-pawsx-qmc-default,xiaobu#output/training-xiaobu-qmc-default,qbqtc#output/training-qbqtc-qmc-default
eval_list=csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,ocnli_dev#${dataset_dir}/ocnli_public/dev.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv,cmnli_dev#${dataset_dir}/cmnli_public/dev.tsv,csnli_dev#${dataset_dir}/csnli_public/dev.tsv

python src/eval.py --model-list ${model_list} --eval-list ${eval_list} > logs/eval.log-zh_sup_general_v1 2>&1
