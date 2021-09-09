import pandas as pd
import itertools
import torch
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
import nlpaug.augmenter.word as naw

# !git clone https://github.com/crim-ca/wiki-bias.git

# zf = ZipFile('/content/wiki-bias/datasets/EN-ft-train.txt.zip')
train = pd.read_csv('/Users/zhoumao/Documents/GitHub/summer-project/datasets/bg/BG-train-unbalanced.txt', sep="\t",
                    names=["label", "sentence"])
dev = pd.read_csv('/Users/zhoumao/Documents/GitHub/summer-project/datasets/bg/BG-dev-balanced.txt', sep="\t",
                  names=["label", "sentence"])
test = pd.read_csv('/Users/zhoumao/Documents/GitHub/summer-project/datasets/bg/BG-test-balanced.txt', sep="\t",
                   names=["label", "sentence"])


# Convert categorical label
def factorize(data):
    data.loc[data.label == '__label__biased', 'label'] = 1  # biased
    data.loc[data.label == '__label__neutral', 'label'] = 0  # neutral
    return data.astype({"label": int})


train = factorize(train)
dev = factorize(dev)
test = factorize(test)

aug = True
if aug:
    def wordaug(data):
        diff = len(data[data['label'] == 1]) - len(data[data['label'] == 0])
        sample = data[data['label'] == 0]['sentence'].sample(n=diff).tolist()
        augmented_text = context_aug.augment(sample)
        sample_data = pd.DataFrame({'sentence': augmented_text, 'label': [0] * len(augmented_text)})
        data = data.append(sample_data)
        return data


    context_aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-multilingual-uncased', action="substitute", aug_p=0.1, device='cpu')
    train = wordaug(train)
    print("word augmentation finished!")

# shuffle the data
train = train.sample(frac=1).reset_index(drop=True)
df = pd.concat([train, dev])

# Get the lists of sentence1, sentence2 and their labels.
df_sentence = df.sentence.values
df_labels = df.label.values

# Hugging Face Trainer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

x_train = list(train["sentence"])
y_train = list(train["label"])

x_val = list(dev["sentence"])
y_val = list(dev["label"])

X_test = list(test["sentence"])
y_test = list(test["label"])

X_train_tokenized = tokenizer(x_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(x_val, padding=True, truncation=True, max_length=512)
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)
test_dataset = Dataset(X_test_tokenized, y_test)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def model_init():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,
        return_dict=True)  # Whether the model returns all hidden-states.
    # attention_probs_dropout_prob=0.5,
    # hidden_dropout_prob=0.5


training_args = TrainingArguments(
    output_dir="test",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    disable_tqdm=True,
    warmup_steps=700,
    weight_decay=0.1,
    load_best_model_at_end=True,
    save_total_limit=1,  # Only last 1 models are saved. Older ones are deleted.
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

best_run = trainer.hyperparameter_search(direction="maximize",
                                         backend="ray")

print(best_run)