from pathlib import Path
import json
import logging
from utils.io import read_flat_file, get_order_from_file, remove_ignore, remove_missing_fprime
from argparse import ArgumentParser

def write_latex_macros(data, args):
    prefix = args.prefix
    s =""
    s += "\\newcommand*\\{}Distance{{{}}}".format(prefix, data['edit']['mean_sum'])
    s += "\\newcommand*\\{}DistanceStd{{{}}}".format(prefix, data['edit']['std_sum'])
    s += "\\newcommand*\\{}DistanceAve{{{}}}".format(prefix, data['edit']['mean_mean'])
    s += "\\newcommand*\\{}DistanceAveStd{{{}}}".format(prefix, data['edit']['std_mean'])

    s += "\\newcommand*\\{}GraphDistance{{{}}}".format(prefix, data['graph_edit']['mean_sum'])
    s += "\\newcommand*\\{}GraphDistanceStd{{{}}}".format(prefix, data['graph_edit']['std_sum'])
    s += "\\newcommand*\\{}GraphDistanceAve{{{}}}".format(prefix, data['graph_edit']['mean_mean'])
    s += "\\newcommand*\\{}GraphDistanceAveStd{{{}}}".format(prefix, data['graph_edit']['std_mean'])

    s += "\\newcommand*\\{}Wer{{{}}}".format(prefix, data['wer']['mean_sum'])
    s += "\\newcommand*\\{}WerStd{{{}}}".format(prefix, data['wer']['std_sum'])
    s += "\\newcommand*\\{}WerAve{{{}}}".format(prefix, data['wer']['mean_mean'])
    s += "\\newcommand*\\{}WerAveStd{{{}}}".format(prefix, data['wer']['std_mean'])

    s += "\\newcommand*\\{}Jaccard{{{}}}".format(prefix, data['jaccard']['mean_sum'])
    s += "\\newcommand*\\{}JaccardStd{{{}}}".format(prefix, data['jaccard']['std_sum'])
    s += "\\newcommand*\\{}JaccardAve{{{}}}".format(prefix, data['jaccard']['mean_mean'])
    s += "\\newcommand*\\{}JaccardAveStd{{{}}}".format(prefix, data['jaccard']['std_mean'])

    s += "\\newcommand*\\{}Semantic{{{}}}".format(prefix, data['semantic']['mean_sum'])
    s += "\\newcommand*\\{}SemanticStd{{{}}}".format(prefix, data['semantic']['std_sum'])
    s += "\\newcommand*\\{}SemanticAve{{{}}}".format(prefix, data['semantic']['mean_mean'])
    s += "\\newcommand*\\{}SemanticAveStd{{{}}}".format(prefix, data['semantic']['std_mean'])

    if args.output:
        with open(args.output, 'w') as F:
            F.write(s)
            logging.info("Written string to %s", args.output)
    else:
        print(s)

    return s

def read_data(path):
    with open(path, "r") as F:
        data = json.load(F)
    return data

def main():
    parser = ArgumentParser()
    parser.add_argument("input", help='file with data')
    parser.add_argument("prefix",
                        help='prefix: this will be part of the macro name')
    parser.add_argument("--output",
                        help="path to latex file if none -> write to terminal")
    args = parser.parse_args()
    #print(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Cannot find: {}, exiting...".format(input_path))

    data = read_data(input_path)
    write_latex_macros(data, args)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
