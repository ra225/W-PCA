# W-PCA
W-PCA Based Gradient-Free Proxy for Efficient Search of Lightweight Language Models, ICLR 2025

## Requirements
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

pip install -r requirements.txt
```

## Download checkpoints
Download the vocabulary file of BERT-base (uncased) from [HERE](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt), and put it into `./pretrained_ckpt/`.  
Download the pre-trained checkpoint of BERT-base (uncased) from [HERE](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin), and put it into `./pretrained_ckpt/`.  


## Prepare dataset
Download the latest dump of Wikipedia from [HERE](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), and extract it into `./dataset/pretrain_data/download_wikipedia/`.  
Download a mirror of BooksCorpus from [HERE](https://t.co/lww3BGREp7?amp=1), and extract it into `./dataset/pretrain_data/download_bookcorpus/`.

### - Pre-training data
```shell
bash create_pretrain_data.sh
bash create_pretrain_feature.sh
```
The features of Wikipedia, BooksCorpus, and their concatenation will be saved into `./dataset/pretrain_data/wikipedia_nomask/`,
`./dataset/pretrain_data/bookcorpus_nomask/`, and `./dataset/pretrain_data/wiki_book_nomask/`, respectively.

### - Fine-tuning data
Download the GLUE dataset using the script in [HERE](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py), and put the files into `./dataset/glue/`.  
Download the SQuAD v1.1 and v2.0 datasets from the following links:  
- [squad-v1.1-train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [squad-v1.1-dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [squad-v2.0-train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [squad-v2.0-dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)

and put them into `./dataset/squad/`.


## Pre-train the supernet
```shell
bash pretrain_supernet.sh
```
The checkpoints will be saved into `./exp/pretrain/supernet/`, 



## Train the teacher model (BERT$_{\rm BASE}$)
```shell
bash train.sh
```
The checkpoints will be saved into `./exp/train/bert_base/`, 
and the names of the sub-directories should be modified into the corresponding task name
(i.e., `mnli`, `qqp`, `qnli`, `sst-2`, `cola`, `sts-b`, `mrpc`, `rte`, `wnli`, `squad1.1`, and `squad2.0`). 
Each sub-directory contains a checkpoint named `best_model.bin`.


## Conduct NAS 
```shell
bash spos.sh
```
The checkpoints will be saved into `./exp/spos/`.


## Distill the student model

```shell
bash spos_finetune_mnli.sh
```
The pre-trained and fine-tuned checkpoints will be saved into 
`./exp/downstream/`.


## Test on the GLUE dataset
```shell
bash test.sh
```
The test results will be saved into `./test_results/`.

## Ranking Evaluation Test
Add W-PCA to [FlexiBERT](https://github.com/aaronserianni/training-free-nas)


## Citation
```
@inproceedings{
wang2025wpca,
title={W-{PCA} Based Gradient-Free Proxy for Efficient Search of Lightweight Language Models},
author={Shang Wang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=YkmbJSHjj7}
}
```
