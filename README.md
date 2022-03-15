# 1. Sentence Embedding of Chinese

**文本表征**（Sentence Embedding）在**文本相似度计算**、**海量文本语义聚类**、**大规模语义检索**、**相似问题匹配**中等工业场景都有重要作用，同时如何学习得到高质量的文本表征模型也在业界和学界有着广泛的研究，这是一个充满各种挑战的问题，比如采用什么模型架构来抽取文本表征向量、采用什么任务来学习文本表征模型、场景有无可用的监督或无监督数据集、离线大模型如何轻量化等。

我们的**出发点**是，基于前沿预训练模型为 backbone，利用开源的语义数据集来训练**通用+专用领域**的文本表征模型，并开源出来供大家在具体业务场景中载入开箱即用。同时也会把这套离线流程做成**一站式可配置化pipeline**，供大家在特定业务场景下自定义学习表征模型。

目前开源中文预训练表征模型概况如下：

| Name | Hub  | Domain | Backbone |
| ---- | ---- | ------ | -------- |
|      |      |        |          |
|      |      |        |          |
|      |      |        |          |

## 1.1 数据集

为了对 like-BERT 预训练模型进行 fine-tune 调优和评测以得到更好的文本表征模，对业界开源的语义相似（STS）、自然语言推理（NLI）、问题匹配（QMC）以及相关性等数据集进行了搜集整理，具体介绍如下：

| 类型           | 数据集                                                       | 简介                                                         | 规模                                               |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| **通用领域**   | [OCNLI](https://www.cluebenchmarks.com/introduce.html)       | 原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。OCNLI为中文语言理解基准测评（CLUE）的一部分。 | **Train**:  50437, **Dev**: 2950                   |
|                | [CMNLI](https://github.com/pluto-junzeng/CNSD)               | 翻译自英文自然语言推理数据集 XNLI 和 MNLI，曾经是中文语言理解基准测评（CLUE）的一部分，现在被 OCNLI 取代。 | **Train**: 391783, **Dev**: 12241                  |
|                | [CSNLI](https://github.com/pluto-junzeng/CNSD)               | 翻译自英文自然语言推理数据集 SNLI。                          | **Train**: 545833, **Dev**: 9314, **Test**: 9176   |
|                | [STS-B-Chinese](https://github.com/pluto-junzeng/CNSD)       | 翻译自英文语义相似数据集 STSbenchmark。                      | **Train**: 5231, **Dev**: 1458, **Test**: 1361     |
|                | [PAWS-X](https://www.luge.ai/#/luge/dataDetail?id=16)        | 释义（含义）匹配数据集，特点是具有高度重叠词汇，重点考察模型对句法结构的理解能力。 | **Train**: 49401, **Dev**: 2000, **Test**: 2000    |
|                | [PKU-Paraphrase-Bank](https://github.com/pkucoli/PKU-Paraphrase-Bank/) | 中文语句复述数据集，也即一句话换种方式描述但语义保持一致。   | 共509832个语句对                                   |
| **问题匹配**   | [LCQMC](https://www.luge.ai/#/luge/dataDetail?id=14)         | 百度知道领域的中文问题匹配大规模数据集，该数据集从百度知道不同领域的用户问题中抽取构建数据。 | **Train**: 238766, **Dev**: 8802, **Test**: 12500  |
|                | [BQCorpus](https://www.luge.ai/#/luge/dataDetail?id=15)      | 银行金融领域的问题匹配数据，包括了从一年的线上银行系统日志里抽取的问题pair对，是目前最大的银行领域问题匹配数据。 | **Train**: 100000, **Dev**: 10000, **Test**: 10000 |
|                | [AFQMC](https://www.cluebenchmarks.com/introduce.html)       | 蚂蚁金服真实金融业务场景中的问题匹配数据集（已脱敏），是中文语言理解基准测评（CLUE）的一部分。 | **Train**: 34334, **Dev**: 4316                    |
|                | [DuQM](https://www.luge.ai/#/luge/dataDetail?id=27)          | 问题匹配评测数据集（label没有公开），属于百度大规模阅读理解数据集（DuReader）的[一部分](https://github.com/baidu/DuReader/tree/master/DuQM)。 | 共50000个语句对                                    |
| **对话和搜索** | [BUSTM: OPPO-xiaobu](https://www.luge.ai/#/luge/dataDetail?id=28) | 通过对闲聊、智能客服、影音娱乐、信息查询等多领域真实用户交互语料进行用户信息脱敏、相似度筛选处理得到，该对话匹配（Dialogue Short Text Matching）数据集主要特点是文本较短、非常口语化、存在文本高度相似而语义不同的难例。 | **Train**: 167173, **Dev**: 10000                  |
|                | [QBQTC](https://github.com/CLUEbenchmark/QBQTC)              | QQ浏览器搜索相关性数据集（QBQTC,QQ Browser Query Title Corpus），是QQ浏览器搜索引擎目前针对大搜场景构建的一个融合了相关性、权威性、内容质量、 时效性等维度标注的学习排序（LTR）数据集，广泛应用在搜索引擎业务场景中。（相关性的含义：0，相关程度差；1，有一定相关性；2，非常相关。） | **Train**: 180000, **Dev**: 20000, **Test**: 5000  |

*以上数据集主要收集整理自[CLUE](https://www.cluebenchmarks.com/introduce.html)（中文语言理解基准评测数据集）、[SimCLUE](https://github.com/CLUEbenchmark/SimCLUE) (整合许多开源文本相似数据集)、[百度千言](https://www.luge.ai/#/)的文本相似度等数据集。*

根据以上收集的数据集构建如下**评测 benchmark**：

| Name                   | Size  | Type          |
| ---------------------- | ----- | ------------- |
| **Chinese-STS-B-dev**  | 1458  | label=0.0~1.0 |
| **Chinese-STS-B-test** | 1361  | label=0.0~1.0 |
| **afqmc-dev**          | 4316  | label=0,1     |
| **lcqmc-dev**          | 8802  | label=0,1     |
| **bqcorpus-dev**       | 10000 | label=0,1     |
| **pawsx_dev**          | 2000  | label=0,1     |
| **oppo-xiaobu-dev**    | 10000 | label=0,1     |

*TODO：目前收集的数据集不论是数量还是多样性都需要进一步扩充以更真实的反映表征模型的效果*

## 1.2 基线模型

选取的基线模型涵盖了主流经典方法，有传统的**词向量技术**、**无监督 BERT** 以及**监督精调 BERT** 等，具体介绍如下：

**（1）Word Embedding**

方案：基于预训练的词向量，对语句所有词向量求平均得到语句的向量。

词向量预训练技术较为成熟，是 NLP 领域第二范式的存在。虽然自从第三范式 BERT 出现以来，词向量技术在诸多任务上效果落后较多，不过由于词向量计算速度快，还算是一个较好的候选基线模型。目前收集到业界开源的预训练词向量有 [fastText](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)、[Tencent Embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html)、[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 等，这里选取了 **fastText** （依赖分词工具 [Stanford Segmenter](https://nlp.stanford.edu/software/segmenter.html)）和 **Tencent Embedding** （取了 small vocab & d=200，依赖分词工具 [jieba](https://github.com/fxsjy/jieba)）作为基于词向量的基线模型。

**（2）Unsupervised BERT**

无监督 BERT 这里主要引用了**两种方案**：一种是 **BERT-mean**，也即对 BERT 编码过后的语句接一个 mean pooling 来的到语句的向量；另一种方式是 [BERT-whitening](https://github.com/bojone/BERT-whitening)（2021.01），该方法对 BERT mean pooling 过后的向量进行白化+降维后处理，[效果](https://kexue.fm/archives/8069)可以追平当时的 SOTA 方案 BERT-flow。

**（3）Supervised BERT**

这里监督 BERT 并非是说在 Chinese-STS-B 训练集上训练过，而是在类似的语义相似数据集上对 BERT 进行过监督调优训练，这里选取了 [SimBERT](https://github.com/ZhuiyiTechnology/simbert) 作为基线模型，相关评测结果摘录自[此处](https://kexue.fm/archives/8321#%E7%BB%93%E6%9E%9C%E6%B1%87%E6%80%BB)（***C-STS-B=67.31, LCQMC=74.42, BQCorpus=45.78, PAWSX=15.33***）。

**（4）以上基线模型效果汇总：**

| Type        | Model          | C-STS-B-dev | C-STS-B-test   | afqmc      | lcqmc                | **bqcorpus**         | **pawsx**       | **xiaobu** |
| ----------- | -------------- | ----------- | -------------- | ---------- | -------------------- | -------------------- | --------------- | ---------- |
| 词向量      | fastText       | 44.76       | 42.55          | 6.54%      | 50.08%               | 23.75%               | 6.55%           | 33.36%     |
| 词向量      | Tencent-emb    | 62.51       | 53.07          | 13.73%     | 44.91%               | 41.08%               | 3.64%           | 35.57%     |
| BERT-无监督 | BERT-mean      | 65.86       | 55.17          | 16.93%     | 52.18%               | 34.05%               | 4.65%           | 35.94%     |
| BERT-无监督 | BERT-whitening | **74.96**   | **65.25**      | 21.91%     | 47.26%               | 37.85%               | **7.77%**       | 37.00%     |
| BERT-有监督 | SimBERT        | 73.42       | 64.85 (67.31*) | **27.99%** | **67.18%** (74.42%*) | **47.15%** (45.78%*) | 4.08% (15.33%*) | **41.16%** |

*其中星标表示引用自相关文献结果，其它结果都是我们计算的，这里的 metric = 100 x spearman；SimBERT 由于经过监督训练在多数任务上效果最优*

## 1.3 表征模型

基于当前收集的开源数据集，进行调优训练后得到通用和专用领域模型如下：

| 适用场景            | **csts_dev** | **csts_test** | **afqmc**  | **lcqmc**  | **bqcorpus** | **pawsx**  | **xiaobu** |
| ------------------- | ------------ | ------------- | ---------- | ---------- | ------------ | ---------- | ---------- |
| **通用领域-v1**     | **84.54%**   | **82.17%**    | 23.80%     | 65.94%     | 45.52%       | 11.52%     | 48.51%     |
| **通用领域-v2**     | 77.20%       | 72.60%        | **36.80%** | 76.92%     | 49.63%       | **16.24%** | 63.16%     |
| **开放问题匹配-v1** | 80.90%       | 76.63%        | 34.51%     | **77.06%** | 52.96%       | 12.98%     | 59.48%     |
| **开放对话匹配-v1** | 78.36%       | 74.46%        | 32.18%     | 75.95%     | 44.01%       | 14.50%     | **66.85%** |
| **金融问题匹配-v1** | 77.40%       | 74.55%        | 36.01%     | 75.75%     | **73.25%**   | 11.58%     | 54.76%     |

*模型训练调优具体细节参见[说明文档](./Training.md)。

# 2. Sentence Embedding of English

英文的文本表征模型复现了[Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf)（2019.11）的工作，有以下三种训练方式：

1. 仅使用 NLI 数据（含有[SNLI](https://arxiv.org/abs/1508.05326)和[MNLI](https://arxiv.org/abs/1704.05426)）进行 fine-tune 训练
2. 仅使用 STS 数据（[STSbenchmark](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)）进行 fine-tune 训练
3. 同时使用 NLI 和 STS 数据进行 fine-tune 训练（先 NLI 再 STS 的串行模式）

训练完成后在 STSbenchmark 测试集上计算语句对的模型向量表示相似度跟真实标签之间的 spearman 相关性系数，效果如下：

| Model                  | SBERT | Ours      |
| ---------------------- | ----- | --------- |
| Avg. GloVe embeddings' | 58.02 | 61.54     |
| Avg. BERT embeddings'  | 46.35 | 47.29     |
| SBERT-STSb-base        | 84.67 | 83.30     |
| SBERT-NLI-base         | 77.03 | 77.20     |
| SBERT-NLI-STSb-base    | 85.35 | 84.31     |
| SBERT-NLI-base-v2      | -     | 84.98     |
| SBERT-NLI-STSb-base-v2 | -     | **87.98** |

*其中'表示没有 fine-tune 的 baseline 模型，SBERT 评估结果参见论文中 Table 1 和 Table 2；SBERT-NLI-base-v2和SBERT-NLI-STSb-base-v2是基于MNRL优化目标训练的，效果更好。*

以上表征模型效果评测基于 STS 数据集，[TSDAE](https://arxiv.org/pdf/2104.06979.pdf) 指出仅依赖 STS 评测表征模型效果是不够的，主要原因有如下几点：

1. STS 数据没有特定领域/任务知识，所以无法反映模型在具体领域/任务上的表现（实验发现 STS 上表现好，但某些下游任务表现反而变差）
2. STS 中相似和不相似数据分布较为平衡，但实际应用场景中往往是极为倾斜的
3. STS 评测时需要等价对待相似和不相似数据的排序问题，而实际应用场景可能仅需要考虑少量相似数据

[Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html#model-overview) 目前给出最好的通用领域语句表征模型是 [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)，该模型以微软 MPNet 作为 backbone 在十亿级[大规模语句对的多种数据集](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)上[训练](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/train_script.py)得到，同时还提供了质量尚可的轻量模型 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，具体训练方法参见[文档](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354)（2022.03）。


