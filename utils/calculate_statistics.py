import logging
import json
import re
import copy
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from utils.io import read_flat_file, write_flat_file_in_order, remove_ignore, remove_missing_fprime
from utils.vectorizer import SentenceVectorizer


def write_file(path, data):
    with open(path, 'w') as F:
        json.dump(data, F, indent=2)
        logging.info("written data to file: {}".format(path))




def main():
    parser = ArgumentParser()
    parser.add_argument("input", help='files to be evaluated, separated with ,')
    parser.add_argument("output", help='output .txt file')
    args = parser.parse_args()

    # read all files
    input_paths = args.input.split(",")
    datas = []
    data_len = 0
    num_files = len(input_paths)

    logging.info("Opening files")
    for path in input_paths:
        try:
            data = read_flat_file(path, return_dict=True)

            # I need to know how many elements
            if len(data) > data_len:
                data_len = len(data)
            datas.append(data)
        except IOError as e:
            print(e)
            print("File {} not found".format(path))
            exit()


    # put all info into a np.array
    metrics = {
        "wer": {},
        "jaccard": {},
        "semantic": {},
        "graph_edit": {},
        "norm_graph_edit": {},
        "edit": {},
        "norm_edit": {}
    }


    req_ids = datas[0].keys()
    stats = {id: copy.deepcopy(metrics) for id in req_ids}
    #print(stats)

    for id in req_ids:
        for metric in metrics:
            if metric in datas[0][id]:
                #print("id: {}".format(id))
                #print("metric: {}".format(metric))
                values = np.array([data[id][metric] for data in datas], dtype=float)
                #print(values)
                #print(values.shape)
                stats[id][metric]['mean'] = values.mean()
                stats[id][metric]['std'] = values.std()




    temp_m = copy.deepcopy(metrics)
    #print(temp_m)

    for data in datas:
        for metric in metrics.keys():
            sum = 0
            num_items = 0
            for id in req_ids:
                if metric in data[id]:
                    value = float(data[id][metric])
                    #print(data[id][metric])
                    sum += value
                    num_items += 1
                if not 'sum' in temp_m[metric]:
                    temp_m[metric]['sum'] = []
                if not 'mean' in temp_m[metric]:
                    temp_m[metric]['mean'] = []
            mean = sum / num_items
            temp_m[metric]['sum'].append(sum)
            temp_m[metric]['mean'].append(mean)


    #print(temp_m)
    # write all means and st.dev to file
    #print(stats)

    for metric, d in temp_m.items():
        sums = np.array(d['sum'])
        mean_sum = sums.mean()
        std_sum = sums.std()

        means = np.array(d['mean'])
        mean_mean = means.mean()
        std_mean = means.std()

        if not metric in stats:
            stats[metric] = {}
        stats[metric]['mean_sum'] = mean_sum
        stats[metric]['std_sum'] = std_sum
        stats[metric]['mean_mean'] = mean_mean
        stats[metric]['std_mean'] = std_mean

    write_file(args.output, stats)






if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
