from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser('Arguments for the code')

parser.add_argument('--conll_path', type=str, default="./data/conll-data/excel_data.txt",
                    help='path to input data')
parser.add_argument('--save_path', type=str, default="./data/conll-data/excel-data-formatted.txt",
                    help='path for output data')

args = parser.parse_args()

# Dictionary defining alternative and 'correct' versions of included tags
labels_dict = {'O':'O',
               'B-PER':'B-PERSON','I-PER':'I-PERSON',
               'B-PERSON':'B-PERSON','I-PERSON':'I-PERSON',
               'B-ORG':'B-ORG','I-ORG':'I-ORG',
               'B-LOC':'B-LOC','I-LOC':'I-LOC',
               'B-GPE':'B-GPE','I-GPE':'I-GPE',
               'B-PROD':'B-PRODUCT','I-PROD':'I-PRODUCT',
               'B-PRODUCT':'B-PRODUCT','I-PRODUCT':'I-PRODUCT',
               'B-EVENT':'B-EVENT','I-EVENT':'I-EVENT',
               'B-DATE':'B-DATE','I-DATE':'I-DATE',
               'B-JON':'B-JON','I-JON':'I-JON',
               'B-FIBC':'B-FIBC','I-FIBC':'I-FIBC',
               'B-NORP':'B-NORP','I-NORP':'I-NORP'}

# List of the tags that are keys in labels_dict
labels_list = list(labels_dict.keys())

# Removes extra tags from conll data
def filter_tags(conll_path, save_path):
    file_ = open(conll_path, 'r')
    _, file_extension = os.path.splitext(conll_path)
    # If input file is .tsv, line is split by \t, and if input is .txt file, line is split by space
    split_sign = '\t' if file_extension == '.tsv' else ' '
    lines = file_.readlines()
    tokens = 0
    replaced_tokens = 0
    with open(save_path, 'w') as f:
        for i, line in enumerate(lines):
            if line != '\n':
                split_line = line.split(split_sign)
                label = split_line[1].strip('\n')
                token = split_line[0]
                tokens += 1
                if label not in labels_list:
                    label = 'O'
                    replaced_tokens += 1
                else:
                    label = labels_dict[label]
                new_line = token + ' ' + label + '\n'
                line = new_line
            f.write(line)
    print('Total number of tokens: ', tokens)
    print('Number of replaced tokens: ', replaced_tokens)

def main():
    filter_tags(args.conll_path, args.save_path)

main()
