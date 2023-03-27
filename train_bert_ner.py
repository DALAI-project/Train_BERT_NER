import json
import os
import torch
import argparse
import pandas as pd
import transformers
import spacy
from collections import defaultdict
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.optim import SGD
import seqeval
from seqeval.metrics import classification_report
from  itertools import chain
from collections import Counter

# Big part of the code is taken and modified from
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# and
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=IEnlUbgm8z3B

parser = argparse.ArgumentParser('Arguments for training BERT based NER model')

parser.add_argument('--data_path', type=str, default="./data/",
                    help='path to data')
parser.add_argument('--save_model_path', type=str, default="./model/test_model",
                    help='path where the trained model is saved')
parser.add_argument('--max_len', type=int, default=512,
                    help='Maximum length of data sequence.')
parser.add_argument('--learning_rate', type=float, default=5e-3,
                    help='Model learning rate.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Size of data batch.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used for the data loaders.')
parser.add_argument('--val_test_size', type=float, default=0.1,
                    help='Proportion of data used for validation.')

args = parser.parse_args()

# Use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Assign numeric ids to named entities
labels_to_ids = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-GPE': 7, 'I-GPE':8,
                 'B-PROD': 9, 'I-PROD': 10, 'B-EVENT': 11, 'I-EVENT': 12, 'B-DATE': 13, 'I-DATE': 14, 'B-JON': 15, 'I-JON': 16, 
                 'B-FIBC': 17, 'I-FIBC': 18, 'B-NOPR': 19, 'I-NOPR': 20}
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# Initialize tokenizer and BERT model
tokenizer = BertTokenizerFast.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
model = BertForTokenClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", num_labels=len(labels_to_ids.keys()))
model.to(device)

# Combine annotated excel documents into pandas dataframe
def excels_to_df(path):
    path = Path(path)
    excel_files = list(path.glob('*.xlsx'))
    all_texts = []
    all_tags = []
    all_nested_tags = []
    for i, file in enumerate(excel_files):
        name = file.stem
        # Read excel into dataframe
        df = pd.read_excel(file)
        df = df.astype(str)
        text = ' '.join(df['word'].tolist())
        all_texts.append(text)
        tags = df['tag'].tolist()
        all_tags.append(tags)
        nested_tags = df['nested_tag'].tolist()
        all_nested_tags.append(nested_tags)
    # Dataframe with columns for document text, tags and nested tags
    df = pd.DataFrame(list(zip(all_texts, all_tags, all_nested_tags)), columns =['texts', 'tags', 'nested_tags'])
    
    return df


# Realigns NER labels with BERT tokenization that often splits words into multiple parts
# (for instance 'kaksoiskappaleet' -> 'kaksois', '##kappale', '##et')
def align_label(text, labels, label_all_tokens = True):
    word_ids = text.word_ids()
    previous_word_idx = None
    label_ids = []
   
    for word_idx in word_ids:
        if word_idx is None:
            # Inidices that should be ignored have a label of -100
            label_ids.append(-100)   
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
        previous_word_idx = word_idx
      
    return label_ids


# Pytorch Dataset for creating batched training data
class DataSequence(Dataset):
    def __init__(self, df, max_len, tokenizer):
        tags = df['tags'].values.tolist()
        texts = df['texts'].values.tolist()
        self.max_len = max_len
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = self.max_len, truncation=True, return_tensors="pt") for i in texts]
        self.labels = [align_label(i,j) for i,j in zip(self.texts, tags)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def tags(df):
    # Get count of the number of tags per category
    s = pd.Series(list(chain.from_iterable(df['tags'])))
    print('NER tags in annotation data: \n', s.value_counts())
    # Replace all tags not included in the annotation list with 'O'
    df['tags'] =  df['tags'].apply(lambda x: [tag if tag in list(labels_to_ids.keys()) else 'O' for tag in x])
    s2 = pd.Series(list(chain.from_iterable(df['tags'])))
    print('\n NER tags in cleaned annotation data: ', s2.value_counts())
    n_unique = len(s2.unique())
    return n_unique


def split_data(df, tokenizer):
    # Split data into train, validation and test datasets
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int((1-2*args.val_test_size) * len(df)), int((1-args.val_test_size) * len(df))])

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(df_train.shape))
    print("VALIDATION Dataset: {}".format(df_val.shape))
    print("TEST Dataset: {}".format(df_test.shape))

    # Create train, validation and test datasets
    train_dataset = DataSequence(df_train, args.max_len, tokenizer)
    val_dataset = DataSequence(df_val, args.max_len, tokenizer)
    test_dataset = DataSequence(df_test, args.max_len, tokenizer)
    # Create train, validation and test dataloaders
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

# http://karpathy.github.io/2019/04/25/recipe/
def test_init_loss(model, dataset, idx, num_classes):
    batch_data, batch_labels = dataset[idx]
    input_ids = batch_data["input_ids"].squeeze(1).to(device)
    attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
    labels = batch_labels.to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    print('Initial loss: ', outputs[0].item())
    print('-ln(1/number of classes) = ', -np.log(1/num_classes))

def test_alignment(tokenizer, dataset, idx):
    batch_data, batch_labels = dataset[idx]
    for token, label in zip(tokenizer.convert_ids_to_tokens(batch_data["input_ids"][0]), batch_labels):
        print('{0:10}  {1}'.format(token, label))

# Function for training the model
def train_loop(model, optimizer, train_dataloader, val_dataloader, bs, epochs):
    best_acc = 0
    best_loss = 1000

    tr_acc_history = []
    val_acc_history = []
    val_loss_history = []
    tr_loss_history = []

    n_train = len(train_dataloader.dataset)
    n_val = len(val_dataloader.dataset)

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        
        # Training loop
        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=train_label, return_dict=False)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        # Validation loop
        for val_data, val_label in tqdm(val_dataloader):
            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=val_label, return_dict=False)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        tr_accuracy = total_acc_train / n_train 
        tr_loss = total_loss_train / n_train
        val_accuracy = total_acc_val / n_val
        val_loss = total_loss_val / n_val

        val_acc_history.append(val_accuracy)
        val_loss_history.append(val_loss)
        tr_acc_history.append(tr_accuracy)
        tr_loss_history.append(tr_loss)


        print(
            f'Epochs: {epoch_num + 1} \
            | Loss: {tr_loss: .3f} \
            | Accuracy: {tr_accuracy: .3f} \
            | Val_Loss: {val_loss: .3f} \
            | Accuracy: {val_accuracy: .3f}')


    hist_dict = {'tr_acc': tr_acc_history, 
                'val_acc': val_acc_history, 
                'val_loss': val_loss_history,
                'tr_loss': tr_loss_history}


    return model, hist_dict


# Function for evaluating trained model with test data
def evaluate(model, test_dataloader):
    test_preds, test_labels = [], []
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0

    for data, labels in test_dataloader:
        labels = labels.to(device)
        mask = data['attention_mask'].squeeze(1).to(device)
        input_id = data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_ids=input_id, attention_mask=mask, labels=labels, return_dict=False)
        
        eval_loss += loss.item()

        nb_eval_steps += 1
        nb_eval_examples += labels.size(0)
              
        # compute evaluation accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
        test_labels.extend(labels)
        test_preds.extend(predictions)
        
        tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in test_labels]
    preds = [ids_to_labels[id.item()] for id in test_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    
    # Reports model performance for each tag category
    print(classification_report([labels], [preds], zero_division=1))


def evaluate_one_text(model, sentence):
    """Function for using the trained model to predict NER tags for given input."""
    text = tokenizer(sentence, padding='max_length', max_length = args.max_len, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_label(sentence)).unsqueeze(0).to(device)

    logits = model(input_ids=input_id, attention_mask=mask, labels=None, return_dict=False)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)


def plot_metrics(hist_dict):
    """Function for plotting the training and validation results."""
    epochs = range(1, args.epochs+1)
    plt.plot(epochs, hist_dict['tr_loss'], 'g', label='Training loss')
    plt.plot(epochs, hist_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./results/tr_val_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_acc'], 'g', label='Training accuracy')
    plt.plot(epochs, hist_dict['val_acc'], 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./results/tr_val_acc.jpg', bbox_inches='tight')
    plt.close()


def main():
    # Combine annotation excels into a dataframe
    df = excels_to_df(args.data_path)
    df = df.sample(n = 100)
    n_unique = tags(df)

    print('Number of documents in data: ', len(df))

    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = split_data(df, tokenizer)

    test_init_loss(model, train_dataset, 5, n_unique)

    test_alignment(tokenizer, train_dataset, 5)

    #optimizer = SGD(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    trained_model, hist_dict = train_loop(model, optimizer, train_dataloader, val_dataloader, args.batch_size, args.epochs)

    # Save trained model
    trained_model.save_pretrained(args.save_model_path, from_pt=True)

    # Evaluate model with test data
    evaluate(trained_model, test_dataloader)

    plot_metrics(hist_dict)

    
main()
