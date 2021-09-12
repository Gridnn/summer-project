# summer-project
Wikipedia cross language biased sentence detection
## 1. Copora
### For model training
The method of datasets creating is based on https://github.com/crim-ca/wiki-bias. We coded pre-processing and sentencizer function for Chinese, it can be accessd by using **normalize.py**, **diff.py** and **nlp.py** instead of original files. 
All datasets including three version, which are balance by unchanged sentences, unbalance and balance by text argumentation are provided in folder **datasets**.
### For model detecting
The articles appearing in both languages are extracted. The code is reported in **current_page_generator.ipynb**



## 2. Model trainig
### Requirements
```
pip install -r /path/to/requirements.txt
```
### FastText
Hyper-parameter search for fastText
```
python3 fastText_classification.py -tr <train_file> -d <dev_file> -te <text_file>
# for example
# python3 fastText_HPSearch.py -tr /datasets/bg/BG-train-unbalanced.txt 
#                              -d /datasets/bg/BG-dev-balanced.txt
#                              -te /datasets/bg/BG-test-balanced.txt
```
### BERT
Hyper-parameter search for BERT
```
python3 BERT_HPSearch.py -tr <train_file> -d <dev_file> -te <text_file> -l <xx>
# for example
# !python3 BERT_HPSearch.py -tr /datasets/bg/BG-train-balanced.txt 
#                           -d /datasets/bg/BG-train-unbalanced.txt 
#                           -te /datasets/bg/BG-dev-balanced.txt 
#                           -l bg
```
