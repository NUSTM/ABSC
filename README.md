# Aspect Level Sentiment Analysis

## Introduction

This is the source code of our paper "Zheng S, Xia R. Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention. arXiv preprint arXiv:1802.00892, 2018.".
Meanwhile, we provide the codes of other papers' models which we implement by [Tensorflow](https://tensorflow.org).

## Recent Papers

1. [Effective LSTMs for Target-Dependent Sentiment Classification with Long Short Term Memory](https://arxiv.org/abs/1512.01100)

    Duyu Tang, Bing Qin, Xiaocheng Feng, Ting Liu (COLING 2016, full paper)
2. [Attention-based LSTM for Aspect-level Sentiment Classification](http://www.aclweb.org/anthology/D/D16/D16-1058.pdf)

    Yequan Wang, Minlie Huang, Li Zhao, Xiaoyan Zhu (EMNLP 2016, full paper)
3. [Aspect Level Sentiment Classification with Deep Memory Network](http://arxiv.org/abs/1605.08900)

    Duyu Tang, Bing Qin, Ting Liu (EMNLP 2016, full paper)
4. [Gated Neural Networks for Targeted Sentiment Analysis](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12074/12065)

    Meishan Zhang, Yue Zhang, Duy-Tin Vo (AAAI 2016)
5. [Interactive Attention Networks for Aspect-Level Sentiment Classification](https://arxiv.org/abs/1709.00893)

    Dehong Ma, Sujian Li, Xiaodong Zhang, Houfeng Wang (IJCAI 2017, full paper)
6. [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](http://www.aclweb.org/anthology/D17-1048)

    Peng Chen, Zhongqian Sun, Lidong Bing, Wei Yang (EMNLP 2017, full paper)
7. [**Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention**](https://arxiv.org/abs/1802.00892)
    **Shiliang Zheng, Rui Xia (Our Paper)**


## source code tree

    .
    ├── README.md
    ├── model
    │   ├── lstm.py          Paper 1
    │   ├── tc_lstm.py       Paper 1
    │   ├── td_lstm.py       Paper 1
    │   ├── at_lstm.py       Paper 2
    │   ├── dmn_lstm.py      Paper 3
    │   ├── ian.py           Paper 5
    │   ├── ram.py           Paper 6
    │   ├── lcr.py           Paper 7


## Usage

Usage of codes:

```
Usage: python model/lcr.py  [options]   [parameters]
Options:
        --train_file_path
        --test_file_path
        --embedding_file_path
        --learning_rate
        --batch_size
        --n_iter
        --random_base
        --l2_reg
        --keep_prob1
        --keep_prob2
```

Give the usage of **lcr.py** for example:

```
python model/lcr.py --train_file_path data/absa/laptop/laptop_2014_train.txt
                    --test_file_path data/absa/laptop/laptop_2014_test.txt
                    --embedding_file_path data/absa/laptop/laptop_word_embedding_42b.txt
                    --learning_rate 0.1
                    --batch_size 25
                    --n_iter 50
                    --random_base 0.1
                    --l2_reg 0.00001
                    --keep_prob1 0.5
                    --keep_prob2 0.5
```



