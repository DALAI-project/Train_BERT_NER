from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import os


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
labels_list = list(labels_dict.keys())

# Removes extra tags from Turku data
def format_turku_tags(conll_path, save_path):
    file_ = open(conll_path, 'r')
    lines = file_.readlines()
    replaced_tokens = 0
    with open(save_path, 'w') as f:
        for i, line in enumerate(lines):
            if line != '\n':
                split_line = line.split('\t')
                label = split_line[1].strip('\n')
                token = split_line[0]
                if label not in labels_list:
                    label = 'O'
                    replaced_tokens += 1
                else:
                    label = labels_dict[label]
                new_line = token + ' ' + label + '\n'
                line = new_line
            f.write(line)
    print('Total number of tokens: ', len(lines))
    print('Number of replaced tokens: ', replaced_tokens)
    
    
# Get tag counts from annotation file in conll-format
def count_tags(conll_path):
    file_ = open(conll_path, 'r')
    lines = file_.readlines()
    tags = []
    nested = 0
    for i, line in enumerate(lines):
        if line != '\n':
            line_list = line.split(' ')
            if len(line_list) > 2:
                nested += 1
                for t in line_list[1:]:
                    tags.append(t.strip('\n'))
            else:
                tags.append((line_list[1].strip('\n')))
    print('Total number of tags: ', len(tags))
    print('Nested tags: ', nested)
    unique_tags = set(tags)
    print('Unique tags: ', len(unique_tags))
    counts = Counter(tags)
    print('\nTag counts:')
    for tag in unique_tags:
        print(tag, counts[tag])


# Lists file names and tags for excel files that contain 
# misspelled tags (B_PERSON etc.)
def bad_tags(excel_path):
    path = Path(excel_path)
    input_files = list(path.glob('*.xlsx'))
    print(len(input_files), ' input files')
    res = []
    for input_file in input_files:
        name = input_file.stem
        df = pd.read_excel(input_file)
        df = df.astype(str)
        tags = list(df.iloc[:,1])
        nested = list(df.iloc[:,2])
        bt = []
        for i in range(len(tags)):
            if tags[i] and tags[i] != 'nan':
                if tags[i] not in labels_list:
                    bt.append(tags[i])
            if nested[i] and nested[i] != 'nan':
                if nested[i] not in labels_list:
                    bt.append(nested[i])
        if bt:
            res.append((name, bt))
    return res


# Creates a conll-type text file from multiple excel files
def excel_to_conll(excel_path, output_path):
    # Get list of input file paths
    path = Path(excel_path)
    input_files = list(path.glob('*.xlsx'))
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
                tag = labels_dict[tag]
                tokens.append([token, tag])

        tokens.append([])

    num_segments = sum([1 for token in tokens if not token])
    print("Found %d segments and %d tokens"%(num_segments + 1, len(tokens) - num_segments))

    with open(output_path, "w") as fh:
        fh.write("\n".join(" ".join(token) for token in tokens))
        print("Output file %s"%output_path)


# Format tags (used for Turku OntoNotes Corpus)
def format_tags(conll_path, save_path):
    file_ = open(conll_path, 'r')
    lines = file_.readlines()
    replaced_tokens = 0
    with open(save_path, 'w') as f:
        for i, line in enumerate(lines):
            if line != '\n':
                split_line = line.split('\t')
                label = split_line[1].strip('\n')
                token = split_line[0]
                if label not in labels_list:
                    label = 'O'
                    replaced_tokens += 1
                else:
                    label = labels_dict[label]
                new_line = token + ' ' + label + '\n'
                line = new_line
            f.write(line)
    print('Total number of tokens: ', len(lines))
    print('Number of replaced tokens: ', replaced_tokens)


def conll_to_segments(filename):
    """
    Convert CoNLL files to segments. This return list of segments and each segment is
    a list of tuples (token, tag)
    :param filename: Path
    :return: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
    """
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token.strip():
                segments.append(segment)
                segment = list()
            else:
                parts = token.split()
                token = (parts[0], parts[1:])
                segment.append(token)

        segments.append(segment)

    return segments

# Split conll-annotations into train, validation and test sets
# Code from ArabicNER (arabiner->bin->process.py)
def train_dev_test_split(input_files, output_path, train_ratio, dev_ratio):
    segments = list()
    filenames = ["train.txt", "val.txt", "test.txt"]

    for input_file in input_files:
        segments += conll_to_segments(input_file)

    n = len(segments)
    np.random.shuffle(segments)
    datasets = np.split(segments, [int(train_ratio*n), int((train_ratio+dev_ratio)*n)])

    # write data to files
    for i in range(len(datasets)):
        filename = os.path.join(output_path, filenames[i])

        with open(filename, "w") as fh:
            text = "\n\n".join(["\n".join([f"{token[0]} {' '.join(token[1])}" for token in segment]) for segment in datasets[i]])
            fh.write(text)
            print("Output file %s", filename)


# Joins multiple annotation files in conll-format into one text file
def combine_conll_files(output_path, input_paths):    
    with open(output_path, 'w') as outfile:
        for i, fname in enumerate(input_paths):
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            if i < len(input_paths)-1:
                outfile.write('\n')
    print('Combined files saved to ', output_path)


def main():
    #excel_path = '/home/ubuntu/NER/ArabicNER/data/excels/diaari'
    #bt = bad_tags(excel_path)
    #print(bt)

    #conll_path = '/home/ubuntu/NER/ArabicNER/data/conll/conll_data.txt'
    #excel_to_conll(excel_path, conll_path)
    #count_tags(conll_path)

    #conll_files = [conll_path, 
    #               '/home/ubuntu/NER/ArabicNER/data/turku/train_formatted.txt',
    #               '/home/ubuntu/NER/ArabicNER/data/turku/val_formatted.txt', 
    #               '/home/ubuntu/NER/ArabicNER/data/turku/test_formatted.txt']
    #combined_conll_path = '/home/ubuntu/NER/ArabicNER/data/conll/conll_data_combined.txt'
    #combine_conll_files(output_path, conll_files)

    #input_files = [combined_conll_path]
    #output_path = '/home/ubuntu/NER/ArabicNER/data/tr_val_test/'
    #train_ratio = 0.8 
    #dev_ratio = 0.1
    #train_dev_test_split(input_files, output_path, train_ratio, dev_ratio)

    #count_tags('/home/ubuntu/NER/ArabicNER/data/tr_val_test/test.txt')
