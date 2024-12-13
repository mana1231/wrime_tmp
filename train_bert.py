from datasets import Dataset, DatasetDict
import pandas as pd
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import argparse

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", required=True, help="path to train json")
    parser.add_argument("-valid", required=True, help="path to valid json")
    parser.add_argument("-epoch", type=int, help="num epoch", default=50)
    parser.add_argument("-batch", type=int, help="batch size", default=64)
    parser.add_argument("-lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("-ms", help="model save name", default='model_save') 
    return parser.parse_args()

def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def make_dataset(json_file):
    df = pd.DataFrame(load_json(json_file))
    df.rename(columns={'writer_sentiment': 'labels'}, inplace=True)
    print(df.head())
    label_mapping = { -2: 0, -1: 1, 0: 2, 1: 3, 2: 4 }
    df['labels'] = df['labels'].map(label_mapping)
    dataset = Dataset.from_pandas(df)
    return dataset

def make_dataset_dict(train, valid):
    dataset = DatasetDict({
        "train": train,
        "validation": valid,
    })
    return dataset

def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-whole-word-masking")
    return tokenizer(batch['sentence'], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")                  # f1-score
    acc = accuracy_score(labels, preds)                               # accuracy
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')       # qwk
    return {"accuracy": acc, "f1": f1, "qwk": qwk}

def main():
    args = load_args()
    # train
    train_json = args.train
    train_dataset = make_dataset(train_json)

    # valid
    valid_json = args.valid
    valid_dataset = make_dataset(valid_json)
    dataset = make_dataset_dict(train_dataset, valid_dataset)

    # model setup
    model_name = "tohoku-nlp/bert-base-japanese-whole-word-masking"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 5
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

    # prepare training
    batch_size = args.batch # args
    training_args = TrainingArguments(
        output_dir="bert",
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        warmup_ratio=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.02,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        logging_strategy="epoch",
        fp16=True,
    )

    # training
    trainer = Trainer(
        model=model, args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

    labels = sorted(set(dataset["train"]["labels"]))

    id2label = {i: str(label) for i, label in enumerate(labels)}
    label2id = {str(label): i for i, label in enumerate(labels)}

    trainer.model.config.id2label = id2label
    trainer.model.config.label2id = label2id
    trainer.save_model(f'bert/{args.ms}') # args

if __name__ == '__main__':
    main()