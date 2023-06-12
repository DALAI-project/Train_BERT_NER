import argparse

parser = argparse.ArgumentParser('Arguments for the code')

parser.add_argument('-l','--list', action='append', help='paths to input files', required=True)
parser.add_argument('--save_path', type=str, default="./data/conll-data/combined_data.txt",
                    help='path for saving combined conll file')

args = parser.parse_args()

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
    combine_conll_files(args.save_path, args.l)

main()
