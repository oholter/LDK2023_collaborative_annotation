import json
from pathlib import Path
from argparse import ArgumentParser

OUTPUT_NAME = ""
OUTPUT_DIR = "annot_files/test"


def to_jsonl(input_file, output_file):
    current_object = {}
    objects = []
    with open(input_file, "r") as input_file:
        for line in input_file:
            line = line.strip().split(": ")
            if len(line) < 2:  # no value
                continue

            key, value = line[0], line[1]
            if len(line) > 2:
                for e in line[2:]:
                    value += " {}".format(e)
            if key == "id":
                if current_object:
                    objects.append(current_object)
                current_object = {"id": value}
            else:
                current_object[key] = value
    objects.append(current_object)
    print("Writing file: {}".format(output_file))
    with open(output_file, "w", encoding='utf8') as output_file:
        for obj in objects:
            output_file.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help="can be a .jsonl file or a dir")
    parser.add_argument("output")

    args = parser.parse_args()

    to_jsonl(args.input, args.output)


if __name__ == '__main__':
    main()
