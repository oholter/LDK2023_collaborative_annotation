from argparse import ArgumentParser
from utils.io import read_flat_file, write_flat_file_in_order
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("output", help="output file")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Cannot find: {}, exiting...".format(input_path))
        exit()

    G = read_flat_file(input_path, return_dict=True)

    for id, g in G.items():
        if 'F' in g:
            g['F'] = ""

    write_flat_file_in_order(Path(args.output), G)



if __name__ == '__main__':
    import logging
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    main()
