import logging
import json
import copy
import openai
import os
import time
import random as rnd
from argparse import ArgumentParser
from pathlib import Path

from utils.io import read_flat_file, write_flat_file_in_order
from utils.prompt_generator import PromptGeneratorFromDictGPT3#, PromptGeneratorFromDictChatGPT



PROMPT_FILE = "prompt.txt"
CONFIG_FILE = "gold_creator/config.json"


CHAT = True


def query_openai(prompt, **kwargs):
    model = kwargs['model']
    temperature = kwargs['temperature']
    stop = kwargs["stop"]
    top_p = kwargs['top_p']
    n = kwargs['n']
    max_tokens = kwargs['max_tokens']

    logging.info("querying OpenAI")

    if CHAT:
        response = openai.ChatCompletion.create(model=model,
                                                messages=[{"role" : "system",
                                                           "content" : prompt}],
                                                temperature=temperature,
                                                top_p = top_p,
                                                n=n,
                                                max_tokens=max_tokens,
                                                stop=stop)
    else:
        response = openai.Completion.create(model=model,
                                            prompt=prompt,
                                            temperature=temperature,
                                            top_p=top_p,
                                            n=n,
                                            stop=stop,
                                            max_tokens=max_tokens)

    return response


def check_consistency(R, G):
    """
    checks that all objects in G are also found in R
    """
    is_consistent = True
    for g in G.keys():
        r = R[g]
        print(r)
        if G[g]['req'] != r['req']:
            logging.warning("G and R are inconsistent")
            print("G[g]: {}".format(G[g]))
            print("r: {}".format(r))
            is_consistent = False
            break

    return is_consistent


def remove_empty_fprime(R):
    """
    removes all elements that do not have an f-prime
    """
    to_ignore = []
    for key, item in R.items():
        if 'F-prime' not in item:
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("id: %d does not have F-prime", idx)
        del R[idx]


def remove_ignore(R):
    """
    removes all items in G marked as ignore,
    no need to remove from R, all elements in G are already removed from R
    """

    to_ignore = []
    for key, item in R.items():
        if 'ignore' in item  \
                and item['ignore'].lower() not in ['false', 'no', '0']:
            to_ignore.append(key)

    for idx in to_ignore:
        logging.info("id: %d marked as IGNORE", idx)
        del R[idx]

    return None





def remove_seen_rs(R, G):
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

def append_to_order(id, path):
    with path.open(mode='a') as F:
        F.write("{}\n".format(id))
        logging.info("appended %d to %s", id, path)


def query_and_write_one(r, G, config, G_path, order_path, id):
    p_gen = PromptGeneratorFromDictGPT3(r, G, **config['prompt'])
    #p_gen = PromptGeneratorFromDictChatGPT(r, G, **config['prompt'])
    prompt = p_gen.generate_prompt()
    new_r = {}
    #print(prompt)
    #input()

    while True:
        try:
            response = query_openai(prompt, **config['openai'])
            logging.info("Got response: {}".format(response))
            # for now, assume only one response
            if CHAT:
                response_text = response['choices'][0]['message']['content'].strip()
            else:
                response_text = response['choices'][0]['text'].strip()

            if response_text:
                logging.info("Response text: {}".format(response_text))
                new_r['id'] = r['id']
                new_r['req'] = r['req']
                new_r['headers'] = r['headers']
                new_r['F'] = response_text
                if 'F-prime' in r:
                    new_r['F-prime'] = r['F-prime']
                break
        except openai.error.APIError as err:
            logging.error("APIError: {}".format(err))
            logging.info("Wait 2 seconds")
            time.sleep(2)
        except openai.error.ServiceUnavailableError as err:
            logging.error("ServiceUnavailableError: {}".format(err))
            logging.info("Wait 2 seconds")
            time.sleep(2)
        except openai.error.APIConnectionError as err:
            logging.error("APIConnectionError: {}".format(err))
            logging.info("Wait 2 seconds")
            time.sleep(2)

    # add requirement to Gold file with the extra fields
    logging.info("Adding id: {} to Gold".format(id))
    G[id] = new_r
    print("new_r: {}".format(new_r))
    new_r = {}

    # write gold file
    write_flat_file_in_order(G_path, G)
    append_to_order(id, order_path)


def main():
    """
    1) reads in file with requirement
    2) reads in file with gold (will be changed)
    3) creates a prompt > prompt.txt
    4) inserts the next requirement into gold-file

    For experiments:
        - Create a new experiment folder
        - Create config.json with experiment setup
        - Create a textual description of the experiment

        - R is a flat file with requirements
            must contain F-prime (correct answer) that will be copied to G
        - G should be an initially empty file (will be created)
    """
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="config file")
    parser.add_argument("--limit", help="max number of iterations", type=int)
    parser.add_argument("--start_with_ids",
                        help="instead of starting with id 0, \
we start with the ids in the prompt->ids in the config.json file",
                        action="store_true")
    args = parser.parse_args()

    if args.cfg:
        config_file = args.cfg
    else:
        config_file = CONFIG_FILE
        logging.warning("Using default config file. Is this what you want?")

    with open(config_file, 'r') as F:
        config = json.load(F)
    io = config['io']
    R_arg = io['flat_file']
    G_arg = io['gold_file']
    order = io['order_file']
    openai.api_key = config['openai']["api_key"]

    # This file should contain all the requirements that we shall use
    # It also contains the F-prime
    R_path = Path(R_arg)
    if not R_path.exists():
        print("Cannot find {}. exiting...".format(R_path))
        exit()

    # This file should either be empty, or it is a growing file (in progress)
    G_path = Path(G_arg)
    if not G_path.exists():
        G_path.touch()
        logging.info("Created gold file %s", G_path)

    order_path = Path(order)
    if not order_path.exists():
        order_path.touch()
        logging.info("Created order file %s", order_path)

    R = read_flat_file(R_path, return_dict=True)
    G = read_flat_file(G_path, return_dict=True)

    G_copy = copy.deepcopy(G)

    # there cannot be any item in G that is not in R
    if not check_consistency(R, G):
        print("R and G are not consistent. Exiting...")
        exit()

    remove_seen_rs(R, G)
    remove_ignore(G)
    remove_ignore(R)

    # For the experiments, we need to remove empty f-prime from R as well
    # Not for gold creation of course
    #
    #remove_empty_fprime(R)

    remove_empty_fprime(G)


    # Used only for the "clustering project"
    processed_ids = 0
    if args.start_with_ids:
        logging.info("Starting with specified ids in config-file")
        ids = config['prompt']['ids']
        for id in ids:
            if (args.limit is not None) and processed_ids >= args.limit:
                print("Reached limit: {}, ending the program".format(args.limit))
                exit()

            if id in R:
                r = R[id]
                query_and_write_one(r, G_copy, config, G_path, order_path, id)
                processed_ids += 1
            else:
                logging.info("id: %d not found in R, already processed?", id)

    # most simple req selection strategy:
    # for each req in R
    #
    for num, (id, r) in enumerate(R.items()):
        if (args.limit is not None) and processed_ids >= args.limit:
            if args.start_with_ids and id in config['prompt']['ids']:
                logging.info("skipping id: {}, already seen".format(id))
                continue
            print("Reached limit: {}, ending the program".format(args.limit))
            break

        query_and_write_one(r, G_copy, config, G_path, order_path, id)
        processed_ids += 1


if __name__ == '__main__':
    main()
