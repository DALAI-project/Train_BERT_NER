from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser('Arguments for the code')

parser.add_argument('--conll_path', type=str, default="./data/conll-data/combined_data.txt",
                    help='path to conll file')
parser.add_argument('--save_path', type=str, default="./data/tr_val_test/",
                    help='path for saving train, validation and test data')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='ratio of training data')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='ratio of validation and test data')
parser.add_argument('--output_type', type=str, default=".csv",
                    help='output file type')                
parser.add_argument('--seed', type=int, default=42,
                    help='seed for numpy random.shuffle')

args = parser.parse_args()

np.random.seed(args.seed)  

def conll_to_segments(filename):
    """
    Convert CoNLL files to segments. This returns list of segments and each segment is
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


def save_csv(datasets, names):
    for i in range(len(datasets)):
        filename = os.path.join(args.save_path, names[i] + '.csv')
        tokens = [[token[0] for token in segment] for segment in datasets[i]]
        tags = [[token[1][0] for token in segment] for segment in datasets[i]]
        df = pd.DataFrame(list(zip(tokens, tags)), columns =['tokens', 'tags'])
        df.to_csv(filename, encoding='utf-8', index=False)
        print("Output file: ", filename)


def save_txt(datasets, names):
    # write data to files
    for i in range(len(datasets)):
        filename = os.path.join(args.save_path, names[i] + '.txt')
        with open(filename, "w") as fh:
            text = "\n\n".join(["\n".join([f"{token[0]} {' '.join(token[1])}" for token in segment]) for segment in datasets[i]])
            fh.write(text)
            print("Output file: ", filename)


# Split conll-annotations into train, validation and test sets
# Code from ArabicNER (arabiner->bin->process.py)
def train_dev_test(input_path, output_path, train_ratio, val_ratio):
    names = ["train", "val", "test"]
    segments = list()
    segments = conll_to_segments(input_path)
    n = len(segments)

    np.random.shuffle(segments)
    segments = np.array(segments, dtype=object)
    datasets = np.split(segments, [int(train_ratio*n), int((train_ratio+val_ratio)*n)])

    if args.output_type == '.csv':
        save_csv(datasets, names)
    else:
        save_txt(datasets, names)

def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    train_dev_test(args.conll_path, args.save_path, args.train_ratio, args.val_ratio)

main()
