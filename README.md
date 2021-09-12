# summer-project
Wikipedia cross language biased sentence detection
## 1. Copora
### For model training
The method of datasets creating is based on https://github.com/crim-ca/wiki-bias. We coded pre-processing and sentencizer function for Chinese, it can be accessd by using **normalize.py**, **diff.py** and **nlp.py** instead of original files.  
All datasets including three version, which are balance by unchanged sentences, unbalance and balance by text argumentation are provided in [datasets](https://github.com/Gridnn/summer-project/tree/main/datasets).
### For model detecting
The code of generating current Wikipedia pages is reported in [current_page_generator].ipynb(https://github.com/Gridnn/summer-project/blob/main/current_page_generator.ipynb).  
The maps of qid and pid of the articles are provided in foler [wiki_pid_qid_map](https://github.com/Gridnn/summer-project/tree/main/wiki_pid_qid_map). It can be used to find the articles appearing in both languages, which is reported in [example.ipynb](https://github.com/Gridnn/summer-project/blob/main/example.ipynb).



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

### 3. Detection
The detection results for all 9 BERT models, and 1 fastText classification models are provided in folder [detection result](https://github.com/Gridnn/summer-project/tree/main/detection%20result). The results are saved in list. The biased sentence is represented by 1, and neutral sentence is represented by 0. It can be read by coding:
```
import pickle
with open("/detection result/bg/application1.0_bg.txt", "rb") as fp:   # Unpickling
  application_10_bg = pickle.load(fp)
```
