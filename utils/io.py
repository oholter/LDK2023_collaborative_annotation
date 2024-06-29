import json
import logging

def write_flat_file_in_order(path, data):
    """
    Usecase:
        Write a Gold file with an additional element
        Must write in order
    """
    sorted_dict = sorted(data.items(), key=lambda x: x[0])
    with path.open(mode='w') as F:
        for i, d in sorted_dict:
            if "id" in d:
                F.write("id: {}\n".format(d['id']))
            else:
                F.write("id: {}\n".format(i))
            if 'req' in d:
                F.write("req: {}\n".format(d['req']))
            else:
                F.write("req: {}\n".format(d['meta']['org']))
            if 'headers' in d:
                F.write("headers: {}\n".format(d['headers']))
            else:
                F.write("headers: {}\n".format(d['meta']['headers']))
            if 'F' in d:
                F.write("F: {}\n".format(d['F']))
            else:
                F.write("F:\n")
            if 'F-prime' in d:
                F.write("F-prime: {}\n".format(d['F-prime']))
            else:
                F.write("F-prime:\n")
            if 'C' in d:
                F.write("C: {}\n".format(d['C']))
            if 'C-prime' in d:
                F.write("C-prime: {}\n".format(d['C-prime']))
            if 'wer' in d:
                F.write("wer: {}\n".format(d['wer']))
            if 'jaccard' in d:
                F.write("jaccard: {}\n".format(d['jaccard']))
            if 'semantic' in d:
                F.write("semantic: {}\n".format(d['semantic']))
            if 'parts' in d:
                F.write("parts: {}\n".format(d['parts']))
            #else:
                #F.write("wer:\n")
            if 'correct' in d:
                F.write("correct: {}\n".format(d['correct']))
            if 'graph_edit' in d:
                F.write("graph_edit: {}\n".format(d['graph_edit']))
            if 'norm_graph_edit' in d:
                F.write("norm_graph_edit: {}\n".format(d['norm_graph_edit']))
            #else:
                #F.write("correct:\n")
            if 'delta' in d:
                F.write("delta: {}\n".format(d['delta']))
            if 'edit' in d:
                F.write("edit: {}\n".format(d['edit']))
            if 'norm_edit' in d:
                F.write("norm_edit: {}\n".format(d['norm_edit']))
            if 'ter' in d:
                F.write("ter: {}\n".format(d['ter']))
            if 'chrf' in d:
                F.write("chrf: {}\n".format(d['chrf']))
            #else:
                #F.write("delta:\n")
            if 'note' in d:
                F.write("note: {}\n".format(d['note']))
            if 'ignore' in d:
                F.write("ignore: {}\n".format(d['ignore']))
            F.write("\n")
    print("Written file: {}".format(path))



def write_flat_file(path, data):
    """
    Write a flat file with the format:
        id: INT
        req: original requirement sentence
        headers: headers
        dl:
    """
    with path.open(mode='w') as F:
        for i, d in enumerate(data):
            if "id" in d:
                F.write("id: {}\n".format(d['id']))
            else:
                F.write("id: {}\n".format(i))
            if 'req' in d:
                F.write("req: {}\n".format(d['req']))
            else:
                F.write("req: {}\n".format(d['meta']['org']))
            if 'headers' in d:
                F.write("headers: {}\n".format(d['headers']))
            else:
                F.write("headers: {}\n".format(d['meta']['headers']))
            if 'F' in d:
                F.write("F: {}\n".format(d['F']))
            #else:
                #F.write("F:\n")
            if 'F-prime' in d:
                F.write("F-prime: {}\n".format(d['F']))
            #else:
                #F.write("F-prime:\n")
            if 'wer' in d:
                F.write("wer: {}\n".format(d['wer']))
            #else:
                #F.write("wer:\n")
            if 'delta' in d:
                F.write("delta: {}\n".format(d['delta']))
            if 'note' in d:
                F.write("note: {}".format(d['note']))
            if 'ignore' in d:
                F.write("ignore: {}".format(d['ignore']))
            #else:
                #F.write("delta:\n")
            F.write("\n")
    print("Written file: {}".format(path))



def read_flat_file(input_file, return_dict=True):
    current_object = {}
    if return_dict:
        objects = {}
    else:
        objects = []
    with open(input_file, "r") as input_file:
        for line in input_file:
            #print(line)
            line = line.strip().split(": ")
            value = ""
            key = ""
            if len(line) < 1:  # nothing
                continue
            elif len(line) < 2:  # empty value
                key = line[0]
                key = key.replace(":", "")
                #value = ""
            elif len(line) == 2:
                key, value = line[0], line[1]
            elif len(line) > 2:
                key = line[0]
                value = line[1]
                for e in line[2:]:
                    #print("value: {}".format(value))
                    #print("e: {}".format(e))
                    value += " {}".format(e)

            if key == "id":
                # adding last object if exists
                if current_object:
                    if return_dict:
                        objects[current_object['id']] = current_object
                    else:
                        objects.append(current_object)
                try:
                    value = int(value)
                except ValueError as e:
                    logging.error("Error while parsing %s", input_file)
                    print(e)
                    exit()

                current_object = {"id": value}
            else:
                #print("key: {}".format(key))
                current_object[key] = value

    if return_dict:
        if current_object:
            objects[current_object['id']] = current_object
    else:
        if current_object:
            objects.append(current_object)
        objects.sort(key=lambda x: x['id'])
    return objects


def remove_ignore(G):
    """
    removes all items in G marked as ignore
    """
    to_ignore = []
    for key, item in G.items():
        if 'ignore' in item  \
                and item['ignore'].lower() not in ['false', 'no', '0']:
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("Ignoring item with id: %d (marked as Ignore)", idx)
        del G[idx]


def remove_missing_fprime(G):
    """
    removes all items in G that do not have an F-prime field
    """
    to_ignore = []
    for key, item in G.items():
        if 'F-prime' not in item:
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("Ignoring item with id: %d (no F-prime)", idx)
        del G[idx]


def remove_missing_cprime(G):
    """
    removes all items in G that do not have a C-prime field
    """
    to_ignore = []
    for key, item in G.items():
        if 'C-prime' not in item:
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("Ignoring item with id: %d (no C-prime)", idx)
        del G[idx]


def read_jsonl(path):
    """
    input: path (pathlib.Path)
    output: list of dict
    """
    with path.open(mode='r') as F:
        return [json.loads(line) for line in F]


def get_order_from_file(path):
    with path.open(mode='r') as F:
        order = []
        for line in F:
            order.append(int(line.strip()))
        print(order)
    return order


