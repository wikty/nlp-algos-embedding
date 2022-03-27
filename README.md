# 1. Sentence Embedding of Chinese

**文本表征**（Sentence Embedding）在**文本相似度计算**、**海量文本语义聚类**、**大规模语义检索**、**相似问题匹配**中等工业场景都有重要作用，同时如何学习得到高质量的文本表征模型也在业界和学界有着广泛的研究，这是一个充满各种挑战的问题，比如采用什么模型架构来抽取文本表征向量、采用什么任务来学习文本表征模型、场景有无可用的监督或无监督数据集、离线大模型如何轻量化等。

我们的**出发点**是，基于前沿预训练模型为 backbone，利用开源的语义数据集来训练**通用+专用领域**的文本表征模型，并开源出来供大家在具体业务场景中载入开箱即用。同时也会把这套离线流程做成**一站式可配置化pipeline**，供大家在特定业务场景下自定义学习表征模型。

目前开源中文预训练表征模型概况如下：

| Name                | Hub                                                          | Domain       | Backbone          |
| ------------------- | ------------------------------------------------------------ | ------------ | ----------------- |
| 通用领域语义匹配-v1 | [链接](https://huggingface.co/DMetaSoul/sbert-chinese-general-v1) | 开放         | bert-base-chinese |
| 通用领域语义匹配-v2 | [链接](https://huggingface.co/DMetaSoul/sbert-chinese-general-v2) | 开放         | bert-base-chinese |
| 开放问题匹配-v1     | [链接](https://huggingface.co/DMetaSoul/sbert-chinese-qmc-domain-v1) | 问题匹配     | bert-base-chinese |
| 开放对话匹配-v1     | [链接](https://huggingface.co/DMetaSoul/sbert-chinese-dtm-domain-v1) | 对话匹配     | bert-base-chinese |
| 金融问题匹配-v1     | [链接](https://huggingface.co/DMetaSoul/sbert-chinese-qmc-finance-v1) | 金融问题匹配 | bert-base-chinese |

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

| Type        | Model          | C-STS-B-dev | C-STS-B-test     | afqmc      | lcqmc                | **bqcorpus**         | **pawsx**       | **xiaobu** |
| ----------- | -------------- | ----------- | ---------------- | ---------- | -------------------- | -------------------- | --------------- | ---------- |
| 词向量      | fastText       | 44.76%      | 42.55%           | 6.54%      | 50.08%               | 23.75%               | 6.55%           | 33.36%     |
| 词向量      | Tencent-emb    | 62.51%      | 53.07%           | 13.73%     | 44.91%               | 41.08%               | 3.64%           | 35.57%     |
| BERT-无监督 | BERT-mean      | 65.86%      | 55.17%           | 16.93%     | 52.18%               | 34.05%               | 4.65%           | 35.94%     |
| BERT-无监督 | BERT-whitening | **74.96%**  | **65.25%**       | 21.91%     | 47.26%               | 37.85%               | **7.77%**       | 37.00%     |
| BERT-有监督 | SimBERT        | 73.42%      | 64.85 %(67.31%*) | **27.99%** | **67.18%** (74.42%*) | **47.15%** (45.78%*) | 4.08% (15.33%*) | **41.16%** |

*其中星标表示引用自相关文献结果，其它结果都是我们计算的，这里的 metric = 100 x spearman；SimBERT 由于经过监督训练在多数任务上效果最优*

## 1.3 表征模型

基于当前收集的开源数据集，进行调优训练后得到通用和专用领域模型如下（具体训练细节参见[说明文档](./Training.md)）：

| 适用场景            | **csts_dev** | **csts_test** | **afqmc**  | **lcqmc**  | **bqcorpus** | **pawsx**  | **xiaobu** |
| ------------------- | ------------ | ------------- | ---------- | ---------- | ------------ | ---------- | ---------- |
| **通用领域-v1**     | **84.54%**   | **82.17%**    | 23.80%     | 65.94%     | 45.52%       | 11.52%     | 48.51%     |
| **通用领域-v2**     | 77.20%       | 72.60%        | **36.80%** | 76.92%     | 49.63%       | **16.24%** | 63.16%     |
| **开放问题匹配-v1** | 80.90%       | 76.63%        | 34.51%     | **77.06%** | 52.96%       | 12.98%     | 59.48%     |
| **开放对话匹配-v1** | 78.36%       | 74.46%        | 32.18%     | 75.95%     | 44.01%       | 14.50%     | **66.85%** |
| **金融问题匹配-v1** | 77.40%       | 74.55%        | 36.01%     | 75.75%     | **73.25%**   | 11.58%     | 54.76%     |

各模型适用场景：

- **通用领域-v1**

  此模型在 NLI、PAWS-X、PKU-Paraphrase-Bank、STS 等语义相似数据集上进行训练，适用于**通用语义匹配**场景，不过此模型在 Chinese-STS 任务上虽然效果较好，但在其它任务上效果并非最优，存在一定过拟合风险。

- **通用领域-v2**

  此模型在百万级语义相似数据集 SimCLUE 上进行训练，适用于**通用语义匹配**场景，从效果来看该模型在各种任务上**泛化能力更好**。

- **开放问题匹配-v1**

  此模型在百度知道问题匹配数据集（LCQMC）上进行训练调优，适用于**开放领域的问题匹配**场景，比如：（洗澡用什么香皂好？vs 洗澡用什么香皂好），（大连哪里拍婚纱照好点？ vs 大连哪里拍婚纱照比较好），（银行卡怎样挂失？，银行卡丢了怎么挂失啊？）

- **开放对话匹配-v1**

  此模型在 OPPO 手机助手小布对话匹配数据集（BUSTM）上进行训练调优，适用于**开放领域的对话匹配**场景（偏口语化），比如：（哪有好玩的 VS 这附近有什么好玩的地方），（定时25分钟 VS 计时半个小时），（我要听王琦的歌 VS 放一首王琦的歌）

- **金融问题匹配-v1**

  此模型在大规模银行问题匹配数据集（BQCorpus）上进行训练调优，适用于**金融领域的问题匹配**场景，比如：（8千日利息400元? VS 10000元日利息多少钱），（提前还款是按全额计息 VS 还款扣款不成功怎么还款？），（为什么我借钱交易失败 VS 刚申请的借款为什么会失败）

## 1.4 无监督方法

某些业务场景下标注数据极少，通过无监督的方式来调优文本表征也是一种方式，当然无监督相比有监督效果会大打折扣，因此这部分无监督模型不做开源。这里对主流无监督方法进行了实验：

|                       | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg**    |
| --------------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ---------- |
| **通用领域-v1(监督)** | 84.54%       | 82.17%        | 23.80%    | 65.94%    | 45.52%       | 11.52%    | 48.51%     | 51.71%     |
| **通用领域-v2(监督)** | 77.20%       | 72.60%        | 36.80%    | 76.92%    | 49.63%       | 16.24%    | 63.16%     | **56.08%** |
| **BERT-whitening**    | 74.96%       | 65.25%        | 21.91%    | 47.26%    | 37.85%       | 7.77%     | 37.00%     | 41.71%     |
| **simcse**            | 74.38%       | 67.55%        | 19.69%    | 52.56%    | 47.60%       | 4.61%     | 38.76%     | 43.59%     |
| **esimcse**           | 75.85%       | 70.51%        | 20.02%    | 51.39%    | 46.49%       | 4.19%     | 39.73%     | 44.03%     |
| **tsdae**             | 72.89%       | 68.07%        | 25.01%    | 57.43%    | 45.68%       | 4.64%     | 40.65%     | 44.91%     |
| **mlm**               | 69.16%       | 60.08%        | 21.59%    | 57.55%    | 37.27%       | 4.19%     | 36.76%     | 40.94%     |
| **ct**                | 75.64%       | 68.49%        | 20.91%    | 57.95%    | 45.38%       | 6.48%     | 40.52%     | **45.05%** |
| **ct2**               | 77.01%       | 70.77%        | 20.02%    | 53.55%    | 48.34%       | 5.13%     | 38.50%     | 44.76%     |

*这里无监督语料数据来自 SimCLUE triplet 训练集合；可见无监督方法虽然比 BERT-whitening 这种后处理方法效果好，但对比开箱即用的有监督方法有 +10% 差距（45.05%->56.08%）*

**无监督适用场景以及使用方式**：

1. 虽然这里的实验表明，无监督方法甚至难以打败开箱即用的有监督方法，但具体到业务场景中还要具体情况具体分析，当没有标注数据时至少无监督方法可以作为尚可接受的 baseline
2. 此外当存在部分标注数据时，可以把无监督作为领域/任务适配的第一阶段预训练，然后将其载入继续在标注数据上有监督训练，[TSDAE]() 论文指出通过无监督预训练对最终下游有监督任务有一定程度提升作用

## 1.5 轻量模型

离线训练好的大模型如果直接用于线上推理，对计算资源有苛刻的需求，而且难以满足业务环境对延迟、吞吐量等性能指标的要求，因此需要借助蒸馏、裁剪、量化等离线优化手段来将大模型转换为轻量模型，当然在模型轻量瘦身的过程中难免会带来效果精度的下降，所以我们的目标就是在保证一定效果精确的前提下，尽可能提升轻量模型的精度。

这里我们主要关注**蒸馏技术**对模型效果和性能的影响，对上述监督调优模型进行蒸馏，基于1万条数据测试，GPU设备是V100，batch_size=16，结果如下：

**(1) 通用领域模型-v1**

*性能：*

|            | Teacher               | Student             | Gap   |
| ---------- | --------------------- | ------------------- | ----- |
| Model      | BERT-12-layers (102M) | BERT-4-layers (45M) | 0.44x |
| Cost       | 23s                   | 12s                 | -47%  |
| Latency    | 37ms                  | 20ms                | -46%  |
| Throughput | 422 sentence/s        | 788 sentence/s      | 1.8x  |

*精度:*

|                | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg** |
| -------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ------- |
| **Teacher**    | 84.54%       | 82.17%        | 23.80%    | 65.94%    | 45.52%       | 11.52%    | 48.51%     | 51.71%  |
| **Student**    | 83.39%       | 79.96%        | 20.25%    | 63.39%    | 43.70%       | 7.54%     | 46.91%     | 49.28%  |
| **Gap** (abs.) | -            | -             | -         | -         | -            | -         | -          | -2.43%  |

**(2) 通用领域模型-v2**

*性能：*

|            | Teacher               | Student             | Gap   |
| ---------- | --------------------- | ------------------- | ----- |
| Model      | BERT-12-layers (102M) | BERT-4-layers (45M) | 0.44x |
| Cost       | 23s                   | 12s                 | -47%  |
| Latency    | 38ms                  | 20ms                | -47%  |
| Throughput | 418 sentence/s        | 791 sentence/s      | 1.9x  |

*精度：*

|                | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg** |
| -------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ------- |
| **Teacher**    | 77.19%       | 72.59%        | 36.79%    | 76.91%    | 49.62%       | 16.24%    | 63.15%     | 56.07%  |
| **Student**    | 76.49%       | 73.33%        | 26.46%    | 64.26%    | 46.02%       | 11.83%    | 52.45%     | 50.12%  |
| **Gap** (abs.) | -            | -             | -         | -         | -            | -         | -          | -5.95%  |

**(3) 开放问题匹配-v1**

*性能：*

|            | Teacher               | Student             | Gap   |
| ---------- | --------------------- | ------------------- | ----- |
| Model      | BERT-12-layers (102M) | BERT-4-layers (45M) | 0.44x |
| Cost       | 23s                   | 12s                 | -47%  |
| Latency    | 38ms                  | 20ms                | -47%  |
| Throughput | 421 sentence/s        | 791 sentence/s      | 1.9x  |

*精度：*

|                | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg** |
| -------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ------- |
| **Teacher**    | 80.90%       | 76.62%        | 34.51%    | 77.05%    | 52.95%       | 12.97%    | 59.47%     | 56.35%  |
| **Student**    | 79.89%       | 76.34%        | 27.59%    | 69.26%    | 49.40%       | 9.06%     | 53.52%     | 52.15%  |
| **Gap** (abs.) | -            | -             | -         | -         | -            | -         | -          | -4.2%   |

**(4) 开放对话匹配-v1**

*性能：*

|            | Teacher               | Student             | Gap   |
| ---------- | --------------------- | ------------------- | ----- |
| Model      | BERT-12-layers (102M) | BERT-4-layers (45M) | 0.44x |
| Cost       | 24s                   | 12s                 | -50%  |
| Latency    | 39ms                  | 19ms                | -51%  |
| Throughput | 407 sentence/s        | 815 sentence/s      | 2.0x  |

*精度：*

|                | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg** |
| -------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ------- |
| **Teacher**    | 78.35%       | 74.45%        | 32.17%    | 75.95%    | 44.00%       | 14.50%    | 66.84%     | 55.17%  |
| **Student**    | 77.99%       | 73.95%        | 27.20%    | 67.49%    | 43.90%       | 10.79%    | 58.21%     | 51.36%  |
| **Gap** (abs.) | -            | -             | -         | -         | -            | -         | -          | -3.81%  |

**(5) 金融问题匹配-v1**

*性能：*

|            | Teacher               | Student             | Gap   |
| ---------- | --------------------- | ------------------- | ----- |
| Model      | BERT-12-layers (102M) | BERT-4-layers (45M) | 0.44x |
| Cost       | 23s                   | 12s                 | -47%  |
| Latency    | 38ms                  | 20ms                | -47%  |
| Throughput | 418 sentence/s        | 791 sentence/s      | 1.9x  |

*精度：*

|                | **csts_dev** | **csts_test** | **afqmc** | **lcqmc** | **bqcorpus** | **pawsx** | **xiaobu** | **Avg** |
| -------------- | ------------ | ------------- | --------- | --------- | ------------ | --------- | ---------- | ------- |
| **Teacher**    | 77.40%       | 74.55%        | 36.00%    | 75.75%    | 73.24%       | 11.58%    | 54.75%     | 57.61%  |
| **Student**    | 75.02%       | 71.99%        | 32.40%    | 67.06%    | 66.35%       | 7.57%     | 49.26%     | 52.80%  |
| **Gap** (abs.) | -            | -             | -         | -         | -            | -         | -          | -4.81%  |

**综上**，从 12 层 BERT 蒸馏为 4 层后，模型参数量缩小到 44%，大概 latency 减半、throughput 翻倍、精度下降 3%~6%。要注意虽然层数缩减到 1/3，但由于 embedding 层参数的存在，整个模型参数量缩减必然比 1/3 要高，且词表规模越大则受影响越大。

# 2. Sentence Embedding of English

## 2.1 有监督

英文的文本表征模型复现了[Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf)（2019.11）的工作，基于开源**监督数据集**有以下三种方式训练表征模型：

1. 仅使用 NLI 数据（含有[SNLI](https://arxiv.org/abs/1508.05326)和[MNLI](https://arxiv.org/abs/1704.05426)）进行 fine-tune 训练
2. 仅使用 STS 数据（[STSbenchmark](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)）进行 fine-tune 训练
3. 同时使用 NLI 和 STS 数据进行 fine-tune 训练（先 NLI 再 STS 的串行模式）

训练完成后在 **STSbenchmark 测试集**上计算语句对的模型向量表示相似度跟真实标签之间的 spearman 相关性系数，效果如下：

| Model                  | SBERT    | Ours      |
| ---------------------- | -------- | --------- |
| Avg. GloVe embeddings' | 58.02    | 61.54     |
| Avg. BERT embeddings'  | 46.35    | 47.29     |
| SBERT-STSb-base        | 84.67    | 83.30     |
| SBERT-NLI-base         | 77.03    | 77.20     |
| SBERT-NLI-STSb-base    | 85.35    | 84.31     |
| SBERT-NLI-base-v2      | 83.9     | 84.98     |
| SBERT-NLI-STSb-base-v2 | **87.3** | **87.98** |
| all-mpnet-base-v2      | -        | 83.42     |

*其中'表示没有 fine-tune 的 baseline 模型，SBERT 评估结果参见[论文](https://arxiv.org/pdf/1908.10084.pdf)中 Table 1 和 Table 2；SBERT-NLI-base-v2和SBERT-NLI-STSb-base-v2是基于MNRL优化目标训练的，效果更好，参见[论文](https://arxiv.org/pdf/1908.10084.pdf)中的 Table 11；虽然 [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html#model-overview) 目前给出最好的通用领域语句表征模型是 [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)（在14+各种领域语义数据集上评测），但该模型在 STS 上表现并非最优，这是由于 STS 和其它下游任务之间并非一定正向相关，下文会解释具体原因。*

注：当前英文 SOTA 模型 all-mpnet-base-v2 以微软 MPNet 作为 backbone 在十亿级[大规模语句对的多种数据集](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)上[训练](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/train_script.py)得到，同时还提供了质量尚可的轻量模型 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，具体训练方法参见[文档](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354)（2022.03）。

## 2.2 无监督

此外考虑到某些业务场景可能不存在监督数据，所以这里也尝试了几种经典的**无监督**表征训练方法，在 **STSbenchmark 测试集**上评测效果如下：

| Model                                                        | Paper     | Ours      |
| ------------------------------------------------------------ | --------- | --------- |
| TSDAE '[1](https://arxiv.org/pdf/1908.10084.pdf)             | 66.0      | 67.62     |
| CT '[1](https://arxiv.org/pdf/1908.10084.pdf)                | **73.9**  | 75.62     |
| CT-2                                                         | -         | **79.19** |
| MLM '[1](https://arxiv.org/pdf/1908.10084.pdf)               | 47.3      | 53.69     |
| BERT-flow '[1](https://arxiv.org/pdf/1908.10084.pdf)         | 48.9      | -         |
| SimCSE '[1](https://arxiv.org/pdf/1908.10084.pdf)            | 73.8      | 76.33     |
| SimCSE-BERT-base '[2](https://arxiv.org/pdf/2104.08821.pdf)  | 76.85     | 76.33     |
| ESimCSE-BERT-base '[3](https://arxiv.org/pdf/2109.04380.pdf) | 80.17     | 76.73     |
| SBERT-NLI-base-v2 '[1](https://arxiv.org/pdf/1908.10084.pdf) (nli-sup) | 83.9      | **84.98** |
| SimCSE-NLI-BERT-base '[2](https://arxiv.org/pdf/2104.08821.pdf) (nli-sup) | **84.25** | -         |

*其中 Paper 表示引用自 [TSDAE](https://arxiv.org/pdf/1908.10084.pdf) Table 11 和 [SimCSE](https://arxiv.org/pdf/2104.08821.pdf) Tabel 5；CT-2 是基于双 encoder+MNRL loss实现；ESimCSE 是 SimCSE 的增强版；我们的无监督训练语料跟 SimCSE 保持一致（百万英文维基语料）*

可以看到，对于 STS 任务来说，即便效果最好的无监督方法（CT/SimCSE）也没有开箱即用的预训练有监督（SBERT-NLI-base-v2）效果好，那无监督的意义在哪里，我们直接使用通用的开源有监督模型不就好了吗？TSDAE 做了两组实验：Domain Adapation 和 Pre-training（见 Table2），**Domain Adapation** 实验指出先 in-domain 无监督再利用 general-domain 有监督（如NLI+STS）数据训练效果比开箱即用监督模型好，而 **Pre-training** 进一步指出先 in-domain 无监督再 in-domain 有监督（也即把无监督作为预训练）比单纯有监督效果好。

值得注意的是以上表征模型效果评测基于 STS 数据集，[TSDAE](https://arxiv.org/pdf/2104.06979.pdf) 指出仅依赖 STS 评测表征模型效果是不够的，主要原因有如下几点：

1. STS 数据没有特定领域/任务知识，所以无法反映模型在具体领域/任务上的表现（实验发现 STS 上表现好，但某些下游任务表现反而变差）
2. STS 中相似和不相似数据分布较为平衡，但实际应用场景中往往是极为倾斜的
3. STS 评测时需要等价对待相似和不相似数据的排序问题，而实际应用场景可能仅需要考虑少量相似数据

## 2.3 轻量模型

从 12 层 BERT 蒸馏为 4 层后，模型参数量缩小到 47%，latency 减半、throughput 翻倍、精度下降 1.43%。

|                  | Teacher                       | Student             | Gap    |
| ---------------- | ----------------------------- | ------------------- | ------ |
| Model            | SBERT-NLI-STSb-base-v2 (109M) | BERT-4-layers (52M) | 0.47x  |
| STS-B            | 87.98%                        | 86.55%              | -1.43% |
| Latency (GPU)    | 32ms                          | 17ms                | -47%   |
| Throughput (GPU) | 488 sentence/s                | 913 sentence/s      | 1.8x   |

