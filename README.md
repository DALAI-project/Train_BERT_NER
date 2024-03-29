# BERT NER model training

Code for training Finnish named entity recognition (NER) model based on BERT. TurkuNLP's FinBERT, available at [HuggingFace](https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1), is used as the base model in the implementation.

## Running the code in a virtual environment

#### Create and activate conda environment using the following commands:

`conda create -n ner_env python=3.8`

`conda activate ner_env`

#### Install dependencies listed in the requirements.txt file:

`pip install -r requirements.txt`

## Input data

The code in `train_bert_ner.py` expects the training, validation and test data in separate .csv files, where the first column ("tokens") contains the text content of the document as a word list, while the second column ("tags") contains a list of the corresponding NER tags in IOB (inside-outside-beginning) form. The names of the data files are expected to be `train.csv`, `val.csv` and `test.csv`.

The `data_utils` folder contains several helper functions for processing the training data.

- **excel_to_conll.py**: Transforms annotations in excel files into a text file in conll-format. The code expects the first column of the excel file to contain the token, and second column the corresponding NER tag. 
  -  `excel_path` argument defines the folder where the excel files are located (by default `./data/excels`)
  -  `save_path` argument sets the location for the resulting .txt file (by default `./data/conll-data/`). If the folder does not exist, it is created by the code.
  -  `name` argument sets the name of the resulting file (by default `excel_data.txt`)

- **filter_conll_tags.py**: Replaces all tags in the input file that are not included in the `labels_list` with the 'O' tag. 
  - `conll_path` argument defines the location of the input file (by default `./data/conll-data/excel_data.txt`)
  - `save_path` argument sets the location for the resulting filtered file (by default `./data/conll-data/`). If the folder does not exist, it is created by the code.
  - `name` argument sets the name of the resulting file (by default `excel-data-formatted.txt`)

- **combine_conll_files.py**: Combines multiple annotation files in conll-format into one output file. 
  - `l` argument defines the locations of the input files (for instance `python combine_conll_files.py -l ./data/conll-data/file1.txt -l ./data/conll-data/file2.txt`) 
  - `save_path` argument sets the location for the resulting file (by default `./data/conll-data/`). If the folder does not exist, it is created by the code.
  - `name` argument sets the name of the resulting file (by default `combined_data.txt`)

- **train_val_test_split.py**: Splits the input file (in conll-format) into separate train, validation and test datasets, which are saved either as .txt or as .csv files. 
  - `save_path` argument sets the location for the resulting files (by default `./data/tr_val_test/`). If the folder does not exist, it is created by the code.
  - `conll_path` argument defines the location of the input file (by default `conll_path`)
  - `train_ratio` argument sets the ratio of the input data used for the train dataset
  - `val_ratio` argument sets the ratio of the input data used for the validation dataset (while the remainder is used for test data)
  - `output_type` argument sets the file type for the output (by default `.csv`, otherwise `.txt`)
  - `seed` argument sets the seed value for the numpy random.shuffle function which is used for shuffling the data (by default `42`). 

- **count_tags.py**: Counts the different types of NER tags in the input file. 
  - `conll_path` argument defines the location of the input file (by default `./data/conll-data/combined_data.txt`). 

- **model_to_onnx.py**: Transforms a trained model into .onnx format (find more information [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#export-a-model-to-onnx-with-optimumexportersonnx)).
  - `checkpoint_path` argument defines the folder where the model file is located
  - `onnx_path` argument sets the location for the transformed .onnx model.

## Model output

The model prints training and validation loss, accuracy and F1 score after each training epoch. These are also plotted into three separate plots that are saved in the given location after training has finished. The model also prints loss, accuracy, precision, recall and F1-score for the test set both with and without the 'O' tags, which dominate the data and can therefore distort the recognition results for the other tags. Precision, recall and F1-score are also printed separately for each entity class.

## Training the model

Several model hyperparameters can be provided as arguments when running the code from the command line:

- **data_path**: Path to data files (in .csv format). Default data path is `./data/`.

- **save_model_path**: Path where the trained model and the plots of the training and evaluation metrics are saved. Default folder is `./checkpoint/`.

- **max_len**: Maximum allowed length (in tokens) of single input text. Default value is `512`.

- **learning_rate**: Learning rate for model training. Default value is `0.00002`.

- **gamma**: Defines learning rate decay in the exponential learning rate scheduler. Default value is `0.8`.

- **wd**: Defines weight decay value passed to the AdamW optimizer. Default value is `0.01`.

- **epochs**: Number of training epochs. Default value is `10`.

- **batch_size**: Size of the data batch. Default batch size is `24`.

- **num_workers**: Number of workers used for the data loaders. Default value is `4`.

- **patience**: Number of epochs to train without improvement in selected metric (by default F1 score) before training is aborted. Default value is `2`.

- **freeze_layers**: Defines the number of BERT layers where the weights are frozen during model training. Default value is `0`.

- **double_lr**: Defines whether different learning rates are used for the BERT layer and classification layer weights. If set to True, 
learning_rate defines the lr for the classification layers, while the lr for the BERT layers is lr / 10. Default value is `False`.

- **classifier_dropout**: Sets the dropout value for the classification layers. Default value is `0.3`.

- **num_validate_during_training**: Defines how many times the model is evaluated with validation data during training epoch. Default value is `1`.

- **scheduler**: Defines whether linear learning rate scheduler (`linear`), exponential learning rate scheduler (`exponential`) or no scheduler (`None`) is used. Default value is `linear`.

If default values for the parameters are used, training can be started with

`python train_bert_ner.py`

All parameter values can also be set on the command line with the starting command 

`python train_bert_ner.py --data_path ./data --save_model_path model/test_model --max_len 512 --learning_rate 5e-3 --epochs 5 --batch_size 2 --num_workers 4 --val_test_size 0.1`

## Training the model with GPU(s)

The [Accelerate](https://huggingface.co/docs/accelerate/index) library is used for facilitating model training in a distributed setting.

- For running the code with all available GPUs (with mixed precision disabled), use the command 
`accelerate launch --multi_gpu train_bert_ner.py`

- Training with all available GPUs and float16 mixed precision can be initiated with 
`accelerate launch --multi_gpu --mixed_precision=fp16 train_bert_ner.py`

- Training code can also be restricted to use only one GPU:
`accelerate launch --num_processes=1 train_ner.py`

- In a multi-GPU setting, the specific GPU that is used for training can be defined using GPU id 
`accelerate launch --num_processes=1 --gpu_ids=0 train_bert_ner.py`

- Run the code only with CPU (also in a GPU setting): `accelerate launch --cpu train_bert_ner.py`

- For all possible parameters, see 
`accelerate launch -h`
