# BERT NER

Code for training Finnish named entity recognition (NER) model based on BERT. The Hugging Face library version of TurkuNLP's FinBERT, available at [HuggingFace](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1), is used as the base model in the implementation.

## Model input

The code expects the training data in excel files, where the first column ("word") contains the text content of the document, while the second column ("tag") contains the corresponding NER tags in IOB (inside-outside-beginning) form. In cases where there are multiple tags for a single token, the third column ("nested_tag") contains the rest of the NER tags.

## Model output

The model prints training and validation loss and accuracy after each training epoch. These are also plotted into two separate plots (one for losses, one for accuracy scores) that are saved to the given location after training has finished. The model also prints loss and accuracy for the test set as well as precision, recall and F1-score separately for each class.

## Training the model

Several model hyperparameters can be provided as arguments when running the code from the command line:

- data_path: path to data files (in .xlsx format)

- save_model_path: path where the trained model is saved

- max_len: maximum allowed length of single input text

- learning_rate: learning  rate for model training

- epochs: number of training epochs

- batch_size: size of the data batch

- num_workers: number of workers used for the data loaders

- val_test_size: proportion of data used for validation and testing

`python3 train_bert_ner.py --data_path ./data --save_model_path model/test_model --max_len 512 --learning_rate 5e-3 --epochs 5 --batch_size 2 --num_workers 4 --val_test_size 0.1`
