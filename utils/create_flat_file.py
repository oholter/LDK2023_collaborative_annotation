from pathlib import Path
from argparse import ArgumentParser
from utils.io import write_flat_file, read_jsonl

# OUTPUT_NAME = "flat"
OUTPUT_NAME = ""
OUTPUT_DIR = "flat_files/"




def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="can be a .jsonl file or a dir")

    args = parser.parse_args()
    input = Path(args.input)

    if not input.exists():
        print("file not found: {}".format(input))
        exit()

    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir()

    if input.is_dir():
        for file in input.iterdir():
            data = read_jsonl(file)
            new_file_path = Path("{}/{}{}.txt".format(OUTPUT_DIR,
                                                      OUTPUT_NAME,
                                                      file.stem))
            write_flat_file(new_file_path, data)
    else:

        data = read_jsonl(input)
        new_file_path = Path("{}/{}{}.txt".format(OUTPUT_DIR,
                                                  OUTPUT_NAME,
                                                  input.stem))
        write_flat_file(new_file_path, data)


if __name__ == '__main__':
    main()
