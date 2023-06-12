from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser('Arguments for the code')

parser.add_argument('--excel_path', type=str, default="./data/excels",
                    help='path to excel files')
parser.add_argument('--save_path', type=str, default="./data/conll-data/excel_data.txt",
                    help='path for saving conll file')

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

# Creates a conll-type text file from multiple excel files
def excel2conll(excel_path, output_path):
    # Get list of input file paths
    path = Path(excel_path)
    input_files = list(path.rglob('*.xlsx'))
    print(len(input_files), ' input files')

    tokens = list()
    # transform each .xlsx file separately
    for input_file in input_files:
        df = pd.read_excel(input_file)
        df = df.astype(str)
        
        # Expects the first column of the excel to contain the token,
        # second column to contain the 'main' tag and third column
        # the optional nested tags
        for index, row in df.iterrows():
            token = df.iloc[index,0] 
            tag = df.iloc[index,1] 
            #nested_tags = df.iloc[index,2] 

            if not token or token == 'nan':
                tokens.append([])
                continue

            if not tag or tag not in labels_list:
                tag = 'O'
                    
            else:
                new_tag = labels_dict[tag]
                tokens.append([token, new_tag])

        tokens.append([])

    num_segments = sum([1 for token in tokens if not token])
    print("Found %d segments and %d tokens"%(num_segments + 1, len(tokens) - num_segments))

    with open(output_path, "w") as fh:
        fh.write("\n".join(" ".join(token) for token in tokens))

    print('File was saved to ', output_path)

def main():
    excel2conll(args.excel_path, args.save_path)

main()
