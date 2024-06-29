import logging
import re
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from utils.io import read_flat_file, write_flat_file_in_order
from utils.vectorizer import SentenceVectorizer

vec = SentenceVectorizer()


def cosine_similarity(array1, array2):
    dot_product = np.dot(array1, array2)
    magnitude1 = np.sqrt(np.dot(array1, array1))
    magnitude2 = np.sqrt(np.dot(array2, array2))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity


def semantic_distance(s1, s2):
    s1_vec = vec.embed(s1)
    s2_vec = vec.embed(s2)
    simil = cosine_similarity(s1_vec, s2_vec)
    return simil


def evaluate_one(g):
    f = g['F']
    fprime = g['F-prime']
    score = semantic_distance(f, fprime)
    return score


def evaluate_all(G):
    for key, g in tqdm(G.items()):
        if 'ignore' in g  \
                and g['ignore'].lower() not in ['false', 'no', '0']:
            continue
        # cannot calculate if this is not present
        elif 'F-prime' not in g or 'F' not in g:
            continue
        else:
            score = evaluate_one(g)
            g['semantic'] = score


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
