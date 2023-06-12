# BERT NER

Code for training Finnish named entity recognition (NER) model based on BERT. The Hugging Face library version of TurkuNLP's FinBERT, available at [HuggingFace](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1), is used as the base model in the implementation.

## Input data

The code in `train_bert_ner.py` expects the training, validation and test data in separate .csv files, where the first column ("tokens") contains the text content of the document as a word list, while the second column ("tags") contains a list of the corresponding NER tags in IOB (inside-outside-beginning) form. The names of the data files are expected to be `train.csv`, `val.csv` and `test.csv`.

## Model output

The model prints training and validation loss, accuracy and F1 score after each training epoch. These are also plotted into three separate plots that are saved in the given location after training has finished. The model also prints loss, accuracy, precision, recall and F1-score for the test set both with and without the 'O' tags, which dominate the data and can therefore distort the recognition results for the other tags. Precision, recall and F1-score are also printed separately for each entity class.



## Training the model

## Running the code using accelerate

# for possible parameters, see 'accelerate launch -h'
# Run the code with all available GPUs with mixed precision disabled: 'accelerate launch --multi_gpu train_ner.py'
# Same as above with mixed precision: 'accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_ner.py'
# Use only one GPU: 'accelerate launch --num_processes=1 train_ner.py'
# Use only one GPU defined using GPU id: 'accelerate launch --num_processes=1 --gpu_ids=0 train_ner.py'

Several model hyperparameters can be provided as arguments when running the code from the command line:

- data_path: Path to data files (in .csv format).

- save_model_path: Path where the trained model is saved.

- max_len: Maximum allowed length (in tokens) of single input text.

- learning_rate: Learning rate for model training.

- gamma: Defines learning rate decay in the exponential learning rate scheduler.

- epochs: Number of training epochs.

- batch_size: Size of the data batch.

- num_workers: Number of workers used for the data loaders.

- patience: Number of epochs to train without improvement in selected metric (by default F1 score) before training is aborted.

- freeze_layers: Defines the number of BERT layers where the weights are frozen during model training.

- double_lr: Defines whether different learning rates are used for the BERT layer and classification layer weights. If set to True, 
learning_rate defines the lr for the classification layers, while the lr for the BERT layers is lr / 10.

- classifier_dropout: Sets the dropout value for the classification layers.

- num_validate_during_training: Defines how many times the model is evaluated with validation data during training epoch.

- linear_scheduler: Defines whether linear or exponential learning rate scheduler is used.

`python3 train_bert_ner.py --data_path ./data --save_model_path model/test_model --max_len 512 --learning_rate 5e-3 --epochs 5 --batch_size 2 --num_workers 4 --val_test_size 0.1`
