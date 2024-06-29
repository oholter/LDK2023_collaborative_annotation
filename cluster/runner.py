# read flat_file
# convert to dict
# create embeddings for each sentence
# store the embeddings in a way that it maintains the relationship with the dict
# cluster the embeddings
# identify the element that is closest to the center in each cluster
# return the corresponding dict element


import logging
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from utils.io import read_flat_file, write_flat_file
from utils.vectorizer import SentenceVectorizer
from cluster.cluster import RequirementCluster


def save_clusters(cluster):
    cluster_file = "cluster.txt"
    with open(cluster_file, 'w') as F:
        for label, cluster in cluster.bins.items():
            #sents = [e['req'] for e in cluster]
            if len(cluster) > 1:
                F.write("\n\n")
                F.write("Cluster {}:\n".format(label))
                for e in cluster:
                    F.write("id: {}\n".format(e['id']))
                    F.write("req: {}\n".format(e['req']))
                    F.write("F-prime: {}\n\n".format(e['F-prime']))
        logging.info("Saved clusters to {}".format(cluster_file))




def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    parser.add_argument("--n_clusters", "-n", default=10, type=int)
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print("{} does not extist, exiting...".format(input_path))
        exit()

    data = read_flat_file(input_path, return_dict=False)

    vec = SentenceVectorizer()

    logging.info("Generating sentence vectors")
    for e in tqdm(data):
        e['vec'] = vec(e['req'])


    logging.info("Clustering requirements")
    cluster = RequirementCluster(data, n_clusters=args.n_clusters)
    #cluster.elbow()
    #cluster.display_clusters(include_single=False)

    closest = cluster.closest()

    #print(closest)
    save_clusters(cluster)

    if args.output:
        output_path = Path(args.output)
        write_flat_file(output_path, closest)










if __name__ == '__main__':
    main()
