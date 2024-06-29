"""
This generates a latex longtable with all the values for:
    distance, jaccarc, and semantic values for each requirement

and the sum/average value at the end

"""


from pathlib import Path
import logging
from utils.io import read_flat_file, get_order_from_file, remove_ignore, remove_missing_fprime
from argparse import ArgumentParser


def order_data(data, order):
    if not order:
        return [e for e in data.values()]
    else:
        ordered_data = []
        for idx in order:
            if idx in data:
                element = data[idx]
                ordered_data.append(element)
        return ordered_data


def write_latex(data, order, args):
    latex_path = Path(args.output)
    s = "\\begin{longtable}{ccccc}\\toprule\n\
    id & distance & wer & jaccard & semantic\\\\ \\midrule\n"

    data = order_data(data, order)
    edits = [int(e['edit']) for e in data if 'edit' in e]
    sum_edits = sum(edits)
    ave_edits = sum_edits / len(edits)

    wers = [float(e['wer']) for e in data if 'wer' in e]
    sum_wers = sum(wers)
    ave_wers = sum_wers / len(wers)

    jaccards = [float(e['jaccard']) for e in data if 'jaccard' in e]
    sum_jaccards = sum(jaccards)
    ave_jaccards = sum_jaccards / len(jaccards)

    semantics = [float(e['semantic']) for e in data if 'semantic' in e]
    sum_semantics = sum(semantics)
    ave_semantics = sum_semantics / len(semantics)

    for e in data:
        if 'edit' in e:
            s += "    {} & {} & {} & {} & {} \\\\\n".format(e['id'],
                                                e['edit'],
                                                round(float(e['wer']), 2),
                                                round(float(e['jaccard']), 2),
                                                round(float(e['semantic']), 2))

    s += "    \\midrule\n"

    s += "    sum & {} & {} & {} & {}\\\\\n".format(round(sum_edits, 2),
                                             round(sum_wers, 2),
                                             round(sum_jaccards, 2),
                                             round(sum_semantics, 2))
    s += "    \\midrule\n"
    s += "    average & {} & {} & {} & {}\\\\\n".format(round(ave_edits, 2),
                                                round(ave_wers, 2),
                                                round(ave_jaccards, 2),
                                                round(ave_semantics, 2))
    s += "    \\bottomrule\n\\end{longtable}\n"


    with latex_path.open(mode='w') as F:
        F.write(s)
        logging.info("Written string to %s", latex_path)





def main():
    parser = ArgumentParser()
    parser.add_argument("input", help='file with data')
    parser.add_argument("--output", help="path to latex file",
                        default="output/table.tex")
    parser.add_argument("--order", help="file with order")
    args = parser.parse_args()
    print(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Cannot find: {}, exiting...".format(input_path))

    G = read_flat_file(input_path, return_dict=True)
    remove_ignore(G)
    remove_missing_fprime(G)

    if args.order:
        order_path = Path(args.order)
        if not order_path.exists():
            print("Cannot find file: {}, exiting...".format(input_path))
            exit()
        order = get_order_from_file(order_path)
    else:
        order = None

    write_latex(G, order, args)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
