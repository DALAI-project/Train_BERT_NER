import os
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser('arguments for the code')

parser.add_argument('--input_path', type=str, default="./data/excels/",
                    help='path to input excel files.')

parser.add_argument('--output_path', type=str, default="./data/conll/conll_data.txt",
                    help='path to output conll files')

args = parser.parse_args()

# Get list of input file paths
path = Path(args.input_path)
input_files = list(path.glob('*.xlsx'))
print(len(input_files), ' input files')

# List all 'correct' NER tags allowed in the data
labels_list = ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-GPE','I-GPE','B-PROD','I-PROD','B-EVENT','I-EVENT','B-DATE','I-DATE','B-JON','I-JON','B-FIBC','I-FIBC','B-NOPR','I-NOPR']

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
        nested_tags = df.iloc[index,2] 

        if not token or token == 'nan':
            tokens.append([])

        if not tag or tag not in labels_list:
            tag = 'O'

        elif nested_tags and nested_tags != 'nan': 
            labels = [tag] + nested_tags.split()
            labels = [l if l in labels_list else 'O' for l in labels]
            tokens.append([token] + labels)
                
        else:
            tokens.append([token, tag])

    tokens.append([])

num_segments = sum([1 for token in tokens if not token])
print("Found %d segments and %d tokens"%(num_segments + 1, len(tokens) - num_segments))

with open(args.output_path, "w") as fh:
    fh.write("\n".join(" ".join(token) for token in tokens))
    print("Output file %s"%args.output_path)
