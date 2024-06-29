import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from utils.io import read_flat_file, write_flat_file_in_order

def formula2set(s):
    """
    split the formula into single tokens
    """
    s = re.sub("\\[", " ", s) # just remove square brackets for string/float
    s = re.sub("\\]", "", s)
    s = re.sub("∃", "∃ ", s)
    s = re.sub("∀", "∀ ", s)
    s = re.sub("(≥\\d)", "\\1 ", s)
    s = re.sub("(≤\\d)", "\\1 ", s)
    s = re.sub("(=\\d)", "\\1 ", s)
    s = re.sub("¬", "¬ ", s)
    s = re.sub("[\\(\\)]", "", s)
    tokens = re.split("[ .]", s)
    return set(tokens)


def jaccard_distance(f1, f2):
    set1 = formula2set(f1)
    set2 = formula2set(f2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_index = len(intersection) / len(union)
    jaccard_distance = 1 - jaccard_index

    #print("set1: {}".format(set1))
    #print("set2: {}".format(set2))
    #print("jaccard_index: {}".format(jaccard_index))
    #input()
    return jaccard_index


def evaluate_one(g):
    f = g['F']
    fprime = g['F-prime']
    score = jaccard_distance(f, fprime)
    #print(score)
    return score


def evaluate_all(G):
    for key, g in G.items():
        if 'ignore' in g  \
                and g['ignore'].lower() not in ['false', 'no', '0']:
            continue
        # cannot calculate if this is not present
        elif 'F-prime' not in g or 'F' not in g:
            continue
        else:
            score = evaluate_one(g)
            g['jaccard'] = score


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help='gold file to be evaluated')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Cannot find: {}, exiting...".format(input_path))

    G = read_flat_file(input_path, return_dict=True)
    evaluate_all(G)
    write_flat_file_in_order(input_path, G)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
