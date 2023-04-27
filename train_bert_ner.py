import os
import torch
import argparse
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

# Big part of the code is taken and modified from
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# and
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=IEnlUbgm8z3B

parser = argparse.ArgumentParser('Arguments for training BERT based NER model')

parser.add_argument('--data_path', type=str, default="/home/ubuntu/NER/ArabicNER/data/tr_val_test/",
                    help='path to data')
parser.add_argument('--save_model_path', type=str, default="./model/18042023",
                    help='path where the trained model is saved')
parser.add_argument('--max_len', type=int, default=512,
                    help='Maximum length of data sequence.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Model learning rate.')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='gamma for exponential decay')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Size of data batch.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used for the data loaders.')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs to train without improvement in selected metric.')
parser.add_argument('--freeze_layers', type=int, default=0,
                    help='Number of BERT layers frozen during training.')
parser.add_argument('--double_lr', action='store', type=bool, required=False, default = False, 
                    help='Whether to use different lrs for feature and classifier parts of the model')
parser.add_argument('--classifier_dropout', type=float, default=0.0,
                    help='dropout value for classifier layer')
parser.add_argument('--num_validate_during_training', type=int, default=1,
                    help='How many times to validate during training. Default is 1=at the last step.')

args = parser.parse_args()

# Use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Assign numeric ids to named entities
labels_to_ids = {'O': 0, 'B-PERSON': 1, 'I-PERSON': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-GPE': 7, 'I-GPE': 8,
                 'B-PRODUCT': 9, 'I-PRODUCT': 10, 'B-EVENT': 11, 'I-EVENT': 12, 'B-DATE': 13, 'I-DATE': 14, 'B-JON': 15, 'I-JON': 16, 
                 'B-FIBC': 17, 'I-FIBC': 18, 'B-NORP': 19, 'I-NORP': 20}
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# Initialize tokenizer and BERT model
tokenizer = BertTokenizerFast.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
model = BertForTokenClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", num_labels=len(labels_to_ids.keys()))
if args.classifier_dropout:
    model.config.classifier_dropout = args.classifier_dropout
model.to(device)

# Turn conll data into lists of labels and tokens
def format_conll(conll_path):
    file_ = open(conll_path, 'r')
    lines = file_.readlines()
    all_tokens = []
    tokens = []
    all_tags = []
    tags = []
    for line in lines:
        if line != '\n':
            split_line = line.split(' ')
            tag = split_line[1].strip('\n')
            token = split_line[0]
            tags.append(tag)
            tokens.append(token)
        else:
            all_tokens.append(tokens)
            all_tags.append(tags)
            tokens=[]
            tags=[]
    return all_tokens, all_tags


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

  
# Freeze BERT layers
def freeze_bert_layers():
    if args.freeze_layers:
	    # We freeze here the embeddings of the model
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

        if args.freeze_layers != -1:
	          # if freeze_layers == -1, we only freeze the embedding layer
	          # otherwise we freeze the first `freeze_layers` encoder layers
            for layer in model.bert.encoder.layer[:args.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
                    

# Pytorch Dataset for creating batched training data
class DataSequence(Dataset):
    def __init__(self, tags, texts, max_len, tokenizer):
        self.max_len = max_len
        self.texts = [tokenizer(text, padding='max_length', max_length = self.max_len, truncation=True, return_tensors="pt", is_split_into_words=True) for text in texts]
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


# Format conll data into Pytorch datasets and dataloaders
def get_data(data_path, tokenizer):
    tr_tokens, tr_tags = format_conll(data_path + 'train.txt')
    val_tokens, val_tags = format_conll(data_path + 'val.txt')
    test_tokens, test_tags = format_conll(data_path + 'test.txt')

    print("TRAIN Dataset: {}".format(sum([len(t) for t in tr_tokens])))
    print("VALIDATION Dataset: {}".format(sum([len(t) for t in val_tokens])))
    print("TEST Dataset: {}".format(sum([len(t) for t in test_tokens])))
    print('\n')

    # Create train, validation and test datasets
    train_dataset = DataSequence(tr_tags, tr_tokens, args.max_len, tokenizer)
    val_dataset = DataSequence(val_tags, val_tokens, args.max_len, tokenizer)
    test_dataset = DataSequence(test_tags, test_tokens, args.max_len, tokenizer)

    # Create train, validation and test dataloaders
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last = True)
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
    print('\n')

def test_alignment(tokenizer, dataset, idx):
    batch_data, batch_labels = dataset[idx]
    for token, label in zip(tokenizer.convert_ids_to_tokens(batch_data["input_ids"][0]), batch_labels):
        if label.item() != -100:
            print('{0:10}  {1}'.format(token, ids_to_labels[label.item()]))
    print('\n')

def format_labels(labels, logits):
    # compute evaluation accuracy
    flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
    # only compute accuracy at active labels
    active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
    labels = torch.masked_select(flattened_targets, active_accuracy)
    predictions = torch.masked_select(flattened_predictions, active_accuracy)

    # Change numeric tags to text format
    tags = [ids_to_labels[id.item()] for id in labels]
    preds = [ids_to_labels[id.item()] for id in predictions]

    return tags, preds

def validate(model, val_dataloader):
    model.eval()

    total_acc_val = 0
    total_loss_val = 0

    # Validation loop
    val_preds, val_labels = [], []
    for val_data, val_label in tqdm(val_dataloader):
        val_label = val_label.to(device)
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)
        with torch.no_grad():
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=val_label, return_dict=False)
        labels, predictions = format_labels(val_label, logits)
        val_labels.append(labels)
        val_preds.append(predictions)
        #val_labels += labels
        #val_preds += predictions

        for i in range(logits.shape[0]):
            logits_clean = logits[i][val_label[i] != -100]
            label_clean = val_label[i][val_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
        
        total_loss_val += loss.item()
    
    return total_loss_val, total_acc_val, val_labels, val_preds

def print_accuracies(total_acc_train,total_loss_train,total_acc_val,total_loss_val, n_train, n_val, 
                     val_acc_history, val_loss_history, tr_acc_history,tr_loss_history,epoch_num, val_labels, val_preds):
    tr_accuracy = total_acc_train / (n_train*args.batch_size) 
    tr_loss = total_loss_train / n_train
    val_accuracy = total_acc_val / (n_val*args.batch_size)
    val_loss = total_loss_val / n_val
    
    #writer.add_scalar("Loss/valid", val_loss, epoch_num)
    #writer.add_scalar("Loss/train", tr_loss, epoch_num)
    #accuracies
    #writer.add_scalar("Accuracy/train", tr_accuracy, epoch_num)
    #writer.add_scalar("Accuracy/valid", val_accuracy, epoch_num)
    #f-scores
    #writer.add_scalar("F-score/balanced", f1_score(val_labels, val_preds, average='weighted'), epoch_num)
    
    val_acc_history.append(val_accuracy)
    val_loss_history.append(val_loss)
    tr_acc_history.append(tr_accuracy)
    tr_loss_history.append(tr_loss)

    print(
        f'Epochs: {epoch_num + 1} \
        | Loss: {tr_loss: .3f} \
        | Accuracy: {tr_accuracy: .3f} \
        | Val_Loss: {val_loss: .3f} \
        | Val_accuracy: {val_accuracy: .3f}')

    print('\n')
    print(classification_report(val_labels, val_preds, zero_division=1))
    
    return val_loss

# Function for training the model
def train_loop(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs):
    best_val_loss = 1000
    epoch_n = 0
    timestep = 0

    tr_acc_history = []
    val_acc_history = []
    val_loss_history = []
    tr_loss_history = []

    n_train = len(train_dataloader)
    n_val = len(val_dataloader)
    
    #calculate frac that is used in determining when to validate
    frac, _ = divmod(n_train,args.num_validate_during_training)
    
    #tensorboard logger
    #if not os.path.isdir(args.save_model_path):
    #    os.makedirs(args.save_model_path)
    #writer = SummaryWriter(args.save_model_path)
    no_improvement = 0
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        
        # Training loop
        for b, (train_data, train_label) in enumerate(tqdm(train_dataloader)):
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
            
            # Avoids exploding gradient by doing gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            if timestep % 10 == 0:
                    print("Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Loss %f"%(epoch_num,b,n_train,timestep,optimizer.param_groups[0]['lr'],loss.item()))
            timestep += 1
            
            #do validation num_validate_during_training times
            if (b+1) % frac == 0:
                print(f'validating on iter {b} on epoch {epoch_num}')
                total_loss_val, total_acc_val, val_labels, val_preds = validate(model, val_dataloader)
                val_loss = print_accuracies(total_acc_train,total_loss_train,total_acc_val,total_loss_val, n_train, n_val, 
                            val_acc_history, val_loss_history, tr_acc_history,tr_loss_history,epoch_num, val_labels, val_preds)

                # Saves model if validation accuracy improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    #epoch_n = epoch_num
                    print('Saving model to ', args.save_model_path)
                    # Save trained model
                    model.save_pretrained(args.save_model_path, from_pt=True)
                    #epoch_n = epoch_num
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= args.patience*args.num_validate_during_training:
                    print('Training is aborted as validation loss has not improved')
                    break
                
        scheduler.step()


    hist_dict = {'tr_acc': tr_acc_history, 
                'val_acc': val_acc_history, 
                'val_loss': val_loss_history,
                'tr_loss': tr_loss_history}

    return hist_dict


# Function for evaluating trained model with test data
def evaluate(model, test_dataloader):
    test_preds, test_labels = [], []
    model.eval()
    
    eval_loss, eval_accuracy, eval_accuracy_wo_O = 0, 0, 0
    nb_eval_steps = 0

    for data, labels in tqdm(test_dataloader):
        labels = labels.to(device)
        mask = data['attention_mask'].squeeze(1).to(device)
        input_id = data['input_ids'].squeeze(1).to(device)

        with torch.no_grad():
            loss, logits = model(input_ids=input_id, attention_mask=mask, labels=labels, return_dict=False)
        
        eval_loss += loss.item()

        nb_eval_steps += 1

        tags, predictions = format_labels(labels, logits)
         
        test_labels.append(tags)
        test_preds.append(predictions)
        
        #find indeces to remove 'O' tags for better accuracy calculations
        indeces_with_O =  np.where(np.array(tags)=='O')[0]
        tmp_eval_accuracy_wo_O = accuracy_score(np.delete(tags, indeces_with_O), np.delete(predictions, indeces_with_O))
        tmp_eval_accuracy = accuracy_score(tags, predictions)
        
        #check if all tags are 'O'-tags -> accuracy= nan
        if np.isnan(tmp_eval_accuracy_wo_O):
            eval_accuracy += tmp_eval_accuracy
        else:
            eval_accuracy += tmp_eval_accuracy
            eval_accuracy_wo_O += tmp_eval_accuracy_wo_O

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Test Loss: {eval_loss}")
    print(f"Test Accuracy: {eval_accuracy}")
    print(f"Test Accuracy w/o O-tags: {eval_accuracy_wo_O}")
    
    # Reports model performance for each tag category
    print(classification_report(test_labels, test_preds, zero_division=1))
    print('\n')


def plot_metrics(hist_dict):
    """Function for plotting the training and validation results."""
    epochs = range(1, (args.epochs*args.num_validate_during_training)+1)
    plt.plot(epochs, hist_dict['tr_loss'], 'g', label='Training loss')
    plt.plot(epochs, hist_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Epochs*num_validate')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.save_model_path + '/tr_val_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_acc'], 'g', label='Training accuracy')
    plt.plot(epochs, hist_dict['val_acc'], 'b', label='Validation accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Epochs*num_validate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(args.save_model_path + '/tr_acc.jpg', bbox_inches='tight')
    plt.close()


def main():

    train_dataset, _, _, train_dataloader, val_dataloader, test_dataloader = get_data(args.data_path, tokenizer)
    freeze_bert_layers()
    n_unique_tags = len(list(labels_to_ids.keys()))
    test_init_loss(model, train_dataset, 55, n_unique_tags)
    test_alignment(tokenizer, train_dataset, 55)

    if args.double_lr:
        optimizer = torch.optim.AdamW([{"params": model.bert.parameters(), "lr": args.learning_rate/10},
                        {"params": model.classifier.parameters(), "lr": args.learning_rate}
                        ])
    else:
    	#optimizer = SGD(model.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1, verbose=True)

    # Train the model
    hist_dict = train_loop(model, optimizer, scheduler, train_dataloader, val_dataloader, args.epochs)

    trained_model = BertForTokenClassification.from_pretrained(args.save_model_path, num_labels=n_unique_tags)
    trained_model.to(device)
    
    # Evaluate model with test data
    evaluate(trained_model, test_dataloader)

    plot_metrics(hist_dict)

    
main()
