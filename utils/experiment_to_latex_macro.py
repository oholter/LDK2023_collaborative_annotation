from pathlib import Path
import logging
from utils.io import read_flat_file, get_order_from_file, remove_ignore, remove_missing_fprime
from argparse import ArgumentParser

def write_latex_macros(data, args):
    distances = [int(e['edit']) for e in data.values() if 'edit' in e]
    sum_distance = sum(distances)
    ave_distance = sum_distance / len(distances)

    wers = [float(e['wer']) for e in data.values() if 'wer' in e]
    sum_wers = sum(wers)
    ave_wers = sum_wers / len(wers)

    jaccards = [float(e['jaccard']) for e in data.values() if 'jaccard' in e]
    sum_jaccards = sum(jaccards)
    ave_jaccards = sum_jaccards / len(jaccards)

    semantics = [float(e['semantic']) for e in data.values() if 'semantic' in e]
    sum_semantics = sum(semantics)
    ave_semantics = sum_semantics / len(semantics)

    graph_distances = [int(e['graph_edit']) for e in data.values() if 'graph_edit' in e]
    sum_graph_distance = sum(graph_distances)
    ave_graph_distance = sum_graph_distance / len(graph_distances)


    prefix = args.prefix
    s =""
    s += "\\newcommand*\\{}Distance{{{}}}".format(prefix, sum_distance)
    s += "\\newcommand*\\{}DistanceAve{{{}}}".format(prefix, ave_distance)

    s += "\\newcommand*\\{}GraphDistance{{{}}}".format(prefix, sum_graph_distance)
    s += "\\newcommand*\\{}GraphDistanceAve{{{}}}".format(prefix, ave_graph_distance)

    s += "\\newcommand*\\{}Wer{{{}}}".format(prefix, sum_wers)
    s += "\\newcommand*\\{}WerAve{{{}}}".format(prefix, ave_wers)

    s += "\\newcommand*\\{}Jaccard{{{}}}".format(prefix, sum_jaccards)
    s += "\\newcommand*\\{}JaccardAve{{{}}}".format(prefix, ave_jaccards)

    s += "\\newcommand*\\{}Semantic{{{}}}".format(prefix, sum_semantics)
    s += "\\newcommand*\\{}SemanticAve{{{}}}".format(prefix, ave_semantics)

    if args.output:
        with open(args.output, 'w') as F:
            F.write(s)
            logging.info("Written string to %s", args.output)
    else:
        print(s)

    return s





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

    G = read_flat_file(input_path, return_dict=True)
    remove_ignore(G)
    remove_missing_fprime(G)

    write_latex_macros(G, args)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
