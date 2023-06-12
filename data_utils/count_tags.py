from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser('Arguments for the code')

parser.add_argument('--conll_path', type=str, default="./data/conll-data/combined_data.txt",
                    help='path to conll data')

args = parser.parse_args()

# Get tag counts from annotation file in conll-format
def count_tags(conll_path):
    file_ = open(conll_path, 'r')
    lines = file_.readlines()
    _, file_extension = os.path.splitext(conll_path)
    # If input file is .tsv, line is split by \t, and if input is .txt file, line is split by space
    split_sign = '\t' if file_extension == '.tsv' else ' '
    tags = []
    nested = 0
    for i, line in enumerate(lines):
        if line != '\n':
            line_list = line.split(split_sign)
            if len(line_list) > 2:
                nested += 1
                for t in line_list[1:]:
                    tags.append(t.strip('\n'))
            else:
                tags.append((line_list[1].strip('\n')))
    print('Total number of tags: ', len(tags))
    print('Nested tags: ', nested)
    unique_tags = list(set(tags))
    print('Unique tags: ', len(unique_tags))
    counts = Counter(tags)
    print('\nTag counts:')
    unique_tags.sort()
    for tag in unique_tags:
        print(tag, counts[tag])

def main():
    count_tags(args.conll_path)

main()
