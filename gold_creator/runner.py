import logging
import json
import copy
import openai
import os
import random as rnd
from argparse import ArgumentParser
from pathlib import Path

from utils.io import read_flat_file, write_flat_file_in_order
from utils.prompt_generator import PromptGeneratorFromDictGPT3



PROMPT_FILE = "prompt.txt"
CONFIG_FILE = "gold_creator/config.json"


def query_openai(prompt, **kwargs):
    model = kwargs['model']
    temperature = kwargs['temperature']
    stop = kwargs["stop"]
    top_p = kwargs['top_p']
    n = kwargs['n']
    max_tokens = kwargs['max_tokens']

    response = openai.Completion.create(model=model,
                                        prompt=prompt,
                                        temperature=temperature,
                                        top_p=top_p,
                                        n=n,
                                        #stop=stop,
                                        max_tokens=max_tokens)
    return response


def check_consistency(R, G):
    """
    checks that all objects in G are also found in R
    """
    is_consistent = True
    for g in G.keys():
        r = R[g]
        if G[g]['req'] != r['req']:
            logging.warning("G and R are inconsistent")
            print("G[g]: {}".format(G[g]))
            print("r: {}".format(r))
            is_consistent = False
            break

    return is_consistent

def remove_ignore(G):
    """
    removes all items in G marked as ignore, or does not have F-prime
    no need to remove from R, all elements in G are already removed from R
    """
    to_ignore = []
    for key, item in G.items():
        if 'ignore' in item  \
                and item['ignore'].lower() not in ['false', 'no', '0']:
            #print("marked to ignore")
            to_ignore.append(key)
        elif 'F-prime' not in item:
            #print("no f-prime")
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("Ignoring item with id: %d", idx)
        del G[idx]

    return None





def filter_R(R, G):
    """
    removes all items of R that are also in G

    no return value !
    """

    for key in G.keys():
        if key in R:
            logging.info("Removing item with id: %d", key)
            del R[key]

    return None


def choose_item(R, random=False, id=None):
    """
    Returns one item from R
    either a random item or the first
    """

    if id:
        if id in R:
            return R[id]
        else:
            logging.warning("id: %d not possible", id)
            return None

    keys = list(R.keys())

    if random:
        idx = rnd.choice(keys)
    else:
        keys.sort()
        idx = keys[0]

    return R[idx]


def main():
    """
    1) reads in file with requirement
    2) reads in file with gold (will be changed)
    3) creates a prompt > prompt.txt
    4) inserts the next requirement into gold-file
    """
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    #parser.add_argument("R", help="file with requirements")
    #parser.add_argument("G", help="file with gold")
    parser.add_argument("--id", "-i",
                        help="optional id of requirement you want label",
                        type=int)
    parser.add_argument("--cfg", help="config file")
    args = parser.parse_args()


    if args.cfg:
        config_file = args.cfg
    else:
        config_file = CONFIG_FILE

    with open(config_file, 'r') as F:
        config = json.load(F)
    io = config['io']
    R_arg = io['flat_file']
    G_arg = io['gold_file']
    openai.api_key = config['openai']["api_key"]

    R_path = Path(R_arg)
    if not R_path.exists():
        print("Cannot find {}. exiting...".format(R_path))
        exit()

    G_path = Path(G_arg)
    if not G_path.exists():
        G_path.touch()
        logging.info("Created file %s", G_path)

    R = read_flat_file(R_path, return_dict=True)
    G = read_flat_file(G_path, return_dict=True)
    G_copy = copy.deepcopy(G)

    if not check_consistency(R, G):
        print("R and G are not consistent. Exiting...")
        exit()

    filter_R(R, G)
    remove_ignore(G)

    r = choose_item(R, id=args.id)
    if not r:
        print("empty r, exiting")
        exit()

    #print("r: {}".format(r))
    p_gen = PromptGeneratorFromDictGPT3(r, G, **config['prompt'])
    prompt = p_gen.generate_prompt()
    #print(prompt)


    # dump prompt to file (easier to copy)
    with open(PROMPT_FILE, 'w') as prompt_file:
        prompt_file.write(prompt)
        logging.info("Written prompt to file: {}".format(PROMPT_FILE))


    response = query_openai(prompt, **config['openai'])
    logging.info("Got response: {}".format(response))
    #for event in response:
        #event_text = event["choices"][0]["text"]
        #response_text += event_text

    response_text = ""
    for choice in response['choices']:
        response_text += choice['text']


    logging.info("Response text: {}".format(response_text))
    r['F'] = response_text

    # add requirement to Gold file with the extra fields
    logging.info("Adding id: {} to Gold".format(r['id']))
    G_copy[r['id']] = r

    # write gold file
    write_flat_file_in_order(G_path, G_copy)








if __name__ == '__main__':
    main()
