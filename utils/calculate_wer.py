import logging
from jiwer import wer
from argparse import ArgumentParser
from pathlib import Path

from utils.io import read_flat_file, write_flat_file_in_order


def evaluate_one(g):
    f = g['F']
    fprime = g['F-prime']
    score = wer(fprime, f)
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
            g['wer'] = score






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
