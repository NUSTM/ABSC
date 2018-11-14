# Aspect-based Sentiment Classification

## Introduction

This is the [Tensorflow](https://tensorflow.org) code of TNet

## Related Papers

1. Li, X.; Bing, L.; Lam, W.; and Shi, B. 2018. [Transformation networks for target-oriented sentiment classification](http://aclweb.org/anthology/P18-1087). ACL 2018.


## source code tree

    .
    ├── README.md
    ├── Tnet
    │   ├── __init__.py
    │   ├── layer.py
    │   ├── main.py
    │   ├── nn_utils.py
    │   ├── utils.py


## Usage

Usage of codes:

```
Usage: python Tnet/main.py  [options]   [parameters]
Options:
        --train_file_path
        --test_file_path
        --embedding_file_path
        --learning_rate
        --batch_size
        --n_filter
        --kernels
        --n_hidden
        --n_iter
        --l2_reg
        --type
```

the usage of Tnet :

```
python Tnet/main.py --train_file_path data/absa/laptop/laptop_2014_train.txt
                    --test_file_path data/absa/laptop/laptop_2014_test.txt
                    --embedding_file_path data/absa/laptop/laptop_word_embedding_42b.txt
                    --learning_rate 0.002
                    --batch_size 64
                    --n_filter 100
                    --kernels 3
                    --n_hidden 100
                    --n_iter 50
                    --l2_reg 0.0001
                    --type 'AS'
```



