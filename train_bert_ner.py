import os
import ast
import stat
import torch
import argparse
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler, DataCollatorForTokenClassification

## Running the code using accelerate

# for possible parameters, see 'accelerate launch -h'
# Run the code with all available GPUs with mixed precision disabled: 'accelerate launch --multi_gpu train_ner.py'
# Same as above with mixed precision: 'accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_ner.py'
# Use only one GPU: 'accelerate launch --num_processes=1 train_ner.py'
# Use only one GPU defined using GPU id: 'accelerate launch --num_processes=1 --gpu_ids=0 train_ner.py'

# nohup accelerate launch --num_processes=1 --gpu_ids=1 train_ner.py > bert_runs/09_06_23_e10_linear_lr_b16_lr2e-5/train_log.txt 2>&1 &
# echo $! > bert_runs/09_06_23_e10_linear_lr_b16_lr2e-5/save_pid.txt


parser = argparse.ArgumentParser('Arguments for training BERT NER model')

parser.add_argument('--data_path', type=str, default="./data/",
                    help='path to data')
parser.add_argument('--save_model_path', type=str, default="./checkpoint/",
                    help='path where the trained model is saved')
parser.add_argument('--max_len', type=int, default=512,
                    help='Maximum length of data sequence.')
parser.add_argument('--learning_rate', type=float, default=0.00002, 
                    help='Model learning rate.')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='gamma for exponential decay')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Size of data batch.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used for the data loaders.')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs to continue training without improvement in selected metric.')
parser.add_argument('--freeze_layers', type=int, default=0,
                    help='Number of BERT layers frozen during training.')
parser.add_argument('--double_lr', action='store', type=bool, required=False, default=False, 
                    help='Whether to use different learning rates for feature and classifier parts of the model.')
parser.add_argument('--classifier_dropout', type=float, default=None,
                    help='Dropout value for classifier layer')
parser.add_argument('--num_validate_during_training', type=int, default=1,
                    help='How many times to validate during training. Default is 1=at the last step.')
parser.add_argument('--linear_scheduler', action='store', type=bool, required=False, default=True, 
                    help='Whether to use linear or exponential scheduler.')

args = parser.parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Assign numeric ids to named entities
labels_to_ids = {'O': 0, 'B-PERSON': 1, 'I-PERSON': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-GPE': 7, 'I-GPE': 8,
                 'B-PRODUCT': 9, 'I-PRODUCT': 10, 'B-EVENT': 11, 'I-EVENT': 12, 'B-DATE': 13, 'I-DATE': 14, 'B-JON': 15, 'I-JON': 16, 
                 'B-FIBC': 17, 'I-FIBC': 18, 'B-NORP': 19, 'I-NORP': 20}
# Another dictionary maps numeric ids to labels
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# Initialize tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
# Collator used for building data batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", id2label=ids_to_labels, label2id=labels_to_ids)
# Initialize Accelerator instance
accelerator = Accelerator()

# Set dropout for classifier layers
if args.classifier_dropout:
    model.config.classifier_dropout = args.classifier_dropout

def de_stringify_lists(example):
    """De-stringifies lists of tags and tokens in the datasets."""
    example["tags"] = [ast.literal_eval(l) for l in example['tags']]
    example["tokens"] = [ast.literal_eval(l) for l in example['tokens']]
    return example

def tags_to_ids(example):
    """Transform NER tags into corresponding (numeric) tag ids."""
    example["tags"] = [[labels_to_ids[tag] for tag in l] for l in example['tags']]
    return example

def align_labels_with_tokens(labels, word_ids):
    """Align NER tags with the tokens generated by the tokenizer.
    This is required because the use of subword tokens changes sequence length."""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    """Tokenize input text and align NER tags with
    the resulting token sequence."""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs

def get_data():
    """Function for creating train, validation and test datasets."""
    # Combine the three datasets in .csv form into one DatasetDict
    datasets = load_dataset('csv', data_files={'train': args.data_path + 'train.csv', 'validation': args.data_path + 'val.csv', 'test': args.data_path + 'test.csv'})
    # De-stringify lists of tags and tokens in the datasets
    datasets = datasets.map(de_stringify_lists, batched=True)
    # Change NER tags in all datasets into numeric form (f.ex. B-PERSON -> 1)
    datasets = datasets.map(tags_to_ids, batched=True)

    # Tokenize all 3 dataset using datasets.map and batched=True to speed up the process
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets["train"].column_names)

    # Create PyTorch DataLoaders
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    val_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader

def postprocess(predictions, labels):
    """Clean up and transform labels and predictions into
    text form for evaluation."""
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[ids_to_labels[l] for l in label if l != -100] for label in labels]

    true_predictions = [
            [ids_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

    return true_predictions, true_labels

def gather_predictions(predictions, labels):
    # Necessary to pad predictions and labels for being gathered
    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

    predictions_gathered = accelerator.gather(predictions)
    labels_gathered = accelerator.gather(labels)
    return predictions_gathered, labels_gathered

def validate(model, dataloader):
    """Function for evaluating the model with validation data during training."""
    # Load evaluation metric
    val_seqeval_metric = evaluate.load("seqeval")
    total_loss_val = 0
    val_preds, val_labels = [], []
    # Evaluation loop
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss_val += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predictions_gathered, labels_gathered = gather_predictions(predictions, labels)
        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        # Adds evaluation results from current batch to metric calculations
        val_seqeval_metric.add_batch(predictions=true_predictions, references=true_labels)
        val_preds += true_predictions
        val_labels += true_labels

    # Computes metrics based on batch results
    results = val_seqeval_metric.compute(zero_division=0)
    mean_loss = total_loss_val / len(dataloader)
    print("\nValidation loss %.3f | Validation precision %.3f | Validation recall %.3f | Validation f1 score %.3f | Validation accuracy %.3f\n"%(mean_loss, results['overall_precision'], results['overall_recall'], results['overall_f1'], results["overall_accuracy"]))
    # Reports model performance for each tag category
    print(classification_report(val_labels, val_preds, zero_division=1))

    return results, mean_loss

def get_optimizer(model):
    """Returns optimizer with one learning rate for all model parameters 
    or separate learning rates for BERT layers and classification layers"""
    if args.double_lr:
        optimizer = torch.optim.AdamW(
            [{"params": model.bert.parameters(), "lr": args.learning_rate/10},
            {"params": model.classifier.parameters(), "lr": args.learning_rate}
            ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    return optimizer

def get_scheduler_(optimizer, train_dataloader):
    """Returns linear or exponential scheduler."""
    if args.linear_scheduler:
        scheduler = get_scheduler(
                    "linear",
                    optimizer=optimizer,
                    num_warmup_steps=round(len(train_dataloader)/5),
                    num_training_steps=len(train_dataloader)*args.epochs,
                    )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1, verbose=True)

    return scheduler

def freeze_bert_layers(model):
    """Freezes BERT layer weights."""
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

def train_eval_loop(model, train_dataloader, val_dataloader, optimizer, scheduler):
    """Perform model training and evaluation."""
    n_train = len(train_dataloader)
    timestep = 0
    best_val_f1 = 0
    no_improvement = 0

    tr_acc_history = []
    tr_loss_history = []
    tr_f1_history = []
    val_acc_history = []
    val_loss_history = []
    val_f1_history = []

    # Load evaluation metric
    tr_seqeval_metric = evaluate.load("seqeval")

    # Calculate frac that is used in determining when to validate
    frac, _ = divmod(n_train, args.num_validate_during_training)

    for epoch in range(args.epochs):
        epoch_loss_train = 0
        epoch_loss_val = 0
        epoch_acc_val = 0
        epoch_f1_val = 0

        # Training loop
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            # Clip gradient norm
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Get training metrics for the current batch
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            predictions_gathered, labels_gathered = gather_predictions(predictions, labels)
            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            tr_seqeval_metric.add_batch(predictions=true_predictions, references=true_labels)

            # Save training loss
            epoch_loss_train += loss.item()

            # Print metrics at defined intervals
            if i % 50 == 0:
                    print("Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Train loss %.3f"%(epoch, i, n_train, timestep, optimizer.param_groups[0]['lr'], loss.item()))
            timestep += 1

            optimizer.zero_grad()

            # Do validation num_validate_during_training times
            if (i + 1) % frac == 0:
                print('\nValidating on iter %i on epoch %i'%(i, epoch))
                val_results, val_loss = validate(model, val_dataloader)
                epoch_acc_val += val_results["overall_accuracy"]
                epoch_f1_val += val_results["overall_f1"]
                epoch_loss_val += val_loss

            if args.linear_scheduler:
                scheduler.step()

        if not args.linear_scheduler:
            scheduler.step()

        # Get training metrics for epoch
        tr_results = tr_seqeval_metric.compute(zero_division=0)

        print("\nEpoch %d | Train accuracy %.3f | Train f1 score %.3f\n"%(epoch, tr_results["overall_accuracy"], tr_results["overall_f1"]))

        # Save the results of training and validation metrics
        tr_loss_history.append(epoch_loss_train / n_train)
        tr_acc_history.append(tr_results["overall_accuracy"])
        tr_f1_history.append(tr_results["overall_f1"])
        val_acc_history.append(epoch_acc_val / args.num_validate_during_training)
        val_loss_history.append(epoch_loss_val / args.num_validate_during_training)
        val_f1_history.append(epoch_f1_val / args.num_validate_during_training)

        # Saves model if validation f1 score improves
        if val_f1_history[-1] > best_val_f1:
            print(f'Validation f1 score {val_f1_history[-1]} improved from {best_val_f1}.')
            best_val_f1 = val_f1_history[-1]
            # Save trained model
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.save_model_path, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.save_model_path)
            print('Model saved to ', args.save_model_path)
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= args.patience*args.num_validate_during_training:
            print('Training is aborted as validation f1 score has not improved')
            break

    hist_dict = {'tr_acc': tr_acc_history,
                'tr_f1': tr_f1_history,
                'tr_loss': tr_loss_history,
                'val_acc': val_acc_history,
                'val_f1': val_f1_history,
                'val_loss': val_loss_history}

    return hist_dict

def test_model(model, dataloader):
    """Evaluates trained model with test data."""
    test_preds, test_labels = [], []
    eval_loss = 0
    n_test = len(dataloader)

    # Load evaluation metric
    ts_seqeval_metric = evaluate.load("seqeval")
    ts_seqeval_metric_wo_0 = evaluate.load("seqeval")

    # Evaluation loop
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        eval_loss += loss.item()

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        predictions_gathered, labels_gathered = gather_predictions(predictions, labels)
        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)

        test_preds += true_predictions
        test_labels += true_labels
        
        # Find indices to remove 'O' tags for better accuracy calculations
        flat_labels = sum(true_labels, [])
        flat_predictions = sum(true_predictions, [])
        indices_with_O =  np.where(np.array(flat_labels, dtype=object)=='O')[0]
        true_labels_wo_0 = np.delete(flat_labels, indices_with_O)
        true_predictions_wo_0 = np.delete(flat_predictions, indices_with_O)

        ts_seqeval_metric.add_batch(predictions=true_predictions, references=true_labels)
        ts_seqeval_metric_wo_0.add_batch(predictions=[true_predictions_wo_0], references=[true_labels_wo_0])

    # Computes metrics based on saved batch results
    results = ts_seqeval_metric.compute(zero_division=0)
    results_wo_0 = ts_seqeval_metric_wo_0.compute(zero_division=0)

    test_loss = eval_loss / n_test

    print("\nTest loss %.3f | Test precision %.3f | Test recall %.3f | Test f1 score %.3f | Test accuracy %.3f\n"%(test_loss, results['overall_precision'], results['overall_recall'], results['overall_f1'], results["overall_accuracy"]))
    print("Test precision w/o O-tags %.3f | Test recall w/o O-tags %.3f | Test f1 score w/o O-tags %.3f | Test accuracy w/o O-tags %.3f\n"%(results_wo_0['overall_precision'], results_wo_0['overall_recall'], results_wo_0['overall_f1'], results_wo_0["overall_accuracy"]))

    # Reports model performance for each tag category
    print(classification_report(test_labels, test_preds, zero_division=1))
    print('\n')


def plot_metrics(hist_dict):
    """Function for plotting the training and validation results."""
    epochs = list(range(1, len(hist_dict['tr_loss'])+1))
    plt.plot(epochs, hist_dict['tr_loss'], 'g', label='Training loss')
    plt.plot(epochs, hist_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.save_model_path + '/tr_val_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_acc'], 'g', label='Training accuracy')
    plt.plot(epochs, hist_dict['val_acc'], 'b', label='Validation accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(args.save_model_path + '/tr_val_acc.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_f1'], 'g', label='Training F1 score')
    plt.plot(epochs, hist_dict['val_f1'], 'b', label='Validation F1 score')
    plt.title('Training & Validation F1 scores')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(args.save_model_path + '/tr_val_f1.jpg', bbox_inches='tight')
    plt.close()


def main():
    # Get dataloaders for train, validation and test data
    train_dataloader, val_dataloader, test_dataloader = get_data()
    # Initialize optimizer
    optimizer = get_optimizer(model)
    # Optionally freezes BERT layers during training
    freeze_bert_layers(model)
    # Prepares the model and data for training in a distributed setup (multi-gpu), if available
    ner_model, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_dataloader)
    scheduler = get_scheduler_(optimizer, train_dataloader)
    # Training and evaluation loop returns a dictionary of the metrics
    hist_dict = train_eval_loop(ner_model, train_dataloader, val_dataloader, optimizer, scheduler)
    # Loads the saved model using AutoModelForTokenClassification class
    trained_model = AutoModelForTokenClassification.from_pretrained(args.save_model_path, num_labels=len(list(labels_to_ids.keys())))
    trained_model = accelerator.prepare(trained_model)
    # Tests the performance of the model with the test dataset
    test_model(trained_model, test_dataloader)
    # Plots train and validation metrics
    plot_metrics(hist_dict)

main()
