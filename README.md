# BERT NER

Code for training Finnish named entity recognition (NER) model based on BERT. The Hugging Face library version of TurkuNLP's FinBERT, available at [HuggingFace](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1), is used as the base model in the implementation.

## Input data

The code in `train_bert_ner.py` expects the training, validation and test data in separate .csv files, where the first column ("tokens") contains the text content of the document as a word list, while the second column ("tags") contains a list of the corresponding NER tags in IOB (inside-outside-beginning) form. The names of the data files are expected to be `train.csv`, `val.csv` and `test.csv`.

## Model output

The model prints training and validation loss, accuracy and F1 score after each training epoch. These are also plotted into three separate plots that are saved in the given location after training has finished. The model also prints loss, accuracy, precision, recall and F1-score for the test set both with and without the 'O' tags, which dominate the data and can therefore distort the recognition results for the other tags. Precision, recall and F1-score are also printed separately for each entity class.

## Training the model

Several model hyperparameters can be provided as arguments when running the code from the command line:

- data_path: Path to data files (in .csv format). Default data path is `./data/`.

- save_model_path: Path where the trained model is saved. Default folder is `./checkpoint/`.

- max_len: Maximum allowed length (in tokens) of single input text. Default value is `512`.

- learning_rate: Learning rate for model training. Default value is `0.0002`.

- gamma: Defines learning rate decay in the exponential learning rate scheduler. Default value is `0.8`.

- epochs: Number of training epochs. Default value is `10`.

- batch_size: Size of the data batch. Default batch size is `16`.

- num_workers: Number of workers used for the data loaders. Default value is `4`.

- patience: Number of epochs to train without improvement in selected metric (by default F1 score) before training is aborted. Default value is `2`.

- freeze_layers: Defines the number of BERT layers where the weights are frozen during model training. Default value is `0`.

- double_lr: Defines whether different learning rates are used for the BERT layer and classification layer weights. If set to True, 
learning_rate defines the lr for the classification layers, while the lr for the BERT layers is lr / 10. Default value is `False`.

- classifier_dropout: Sets the dropout value for the classification layers. Default value is `None`.

- num_validate_during_training: Defines how many times the model is evaluated with validation data during training epoch. Default value is `1`.

- linear_scheduler: Defines whether linear or exponential learning rate scheduler is used. Default value is `True`.

If default values for the parameters are used, training can be started with

`python train_bert_ner.py`

All parameter values can also be set on the command line with the starting command 

`python train_bert_ner.py --data_path ./data --save_model_path model/test_model --max_len 512 --learning_rate 5e-3 --epochs 5 --batch_size 2 --num_workers 4 --val_test_size 0.1`

## Training the model with GPU(s)

The [Accelerate](https://huggingface.co/docs/accelerate/index) library is used for facilitating model training in a distributed setting.

For running the code with all available GPUs (with mixed precision disabled), use the command 
`accelerate launch --multi_gpu train_bert_ner.py`

Training with all available GPUs and float16 mixed precision can be initiated with 
`accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train_ner.py`

Training code can also be restricted to use only one GPU
`accelerate launch --num_processes=1 train_ner.py`

In a multi-GPU setting, the specific GPU that is used for training can be defined using GPU id 
`accelerate launch --num_processes=1 --gpu_ids=0 train_ner.py`

For all possible parameters, see 
`accelerate launch -h`
