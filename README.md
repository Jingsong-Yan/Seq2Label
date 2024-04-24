# Does the Order Matter? A Random Generative Way to Learn Label Hierarchy for Hierarchical Text Classification [中文 | [English](./README_EN.md)]

**摘要**: Hierarchical Text Classification (HTC) is an essential and challenging task due to the difficulty of modeling
label hierarchy. Recent generative methods have achieved state-of-the-art performance by flattening the local label
hierarchy intoalabelsequence with a specific order. However, the order between labels does not naturally exist and the
generation of the current label should incorporate the information in all other target labels. Moreover, the generative
methods usually suffer from the error accumulation problem. To this end, we propose a new framework named
sequence-to-label (Seq2Label) with a random generative way to learn label hierarchy for hierarchical text
classification. Instead of using only one specific order, we shuffle the label sequence by a Label Sequence Random
Shuffling (LSRS) mechanism so that a text will be mapped to several different order label sequences during the training
phase. To alleviate the error accumulation problem, we further propose a Hierarchy-aware Negative Sampling (HNS)
strategy with a negative label-aware loss to better distinguish target labels and negative labels. In this way, our
model can capture the hierarchical and co-occurrence information of the target labels of each text. The experimental
results on three benchmark datasets show that Seq2Label achieves state-of-the-art results.
<img src="./images/seq2label.png" align='center'> </div>

## 依赖包

```
pip install requirements.txt
```

## 数据集

- [WOS](https://github.com/kk7nc/HDLTex)
- [RCV1](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)
    - **注意**：真正有用的数据需要从[链接](https://trec.nist.gov/data/reuters/reuters.html)获取。
- [BGC](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)

可以从链接中获取整理好的数据集。[百度网盘链接](https://pan.baidu.com/s/182WLuouZRU5CNivmJaDOHg?pwd=1234 )

每个数据集包括以下文件，`label.taxonomy`、`train.pkl`、`test.pkl`、`valid.pkl`。其中，`label.taxonomy` 保存了标签之间的层次信息。

**数据集格式**(文本, 标签, 稀疏向量)，
其中标签为一个列表，记录了文本所有的标签；稀疏向量是一个标签数长度的向量，向量[标签索引]= 1。

## 训练

1. 在 `config.yaml` 中指定模型路径，默认为 `facebook/bart-base`，
   运行会自动从huggingface下载预训练模型参数。如果存在网络问题，请自行到huggingface中下载参数到本地，并修改为本地路径。
2. 在 `train.sh` 中指定模型 `model`、数据集 `dataset`、标签顺序 `mode`、数据集根目录 `data_path`，使用你配置好的python环境。
3. 执行脚本

```
  bash train.sh
```

## 引用

```
@article{yan2023does,
  title={Does the Order Matter? A Random Generative Way to Learn Label Hierarchy for Hierarchical Text Classification},
  author={Yan, Jingsong and Li, Piji and Chen, Haibin and Zheng, Junhao and Ma, Qianli},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2023},
  publisher={IEEE}
}
```