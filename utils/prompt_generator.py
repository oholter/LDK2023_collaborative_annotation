import logging
import re
import json
import random as rand
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from argparse import ArgumentParser
from utils.io import read_jsonl, read_flat_file, remove_missing_cprime, remove_missing_fprime, remove_ignore
from utils.vectorizer import SentenceVectorizer


MAX_TOKENS = 600


def filter_data(data):
    #return [d for d in data if "F-prime" in d]
    return {g_id: g for g_id, g in data.items() if "F-prime" in g}


def get_req_with_id(path, id):
    data = read_flat_file(path)
    for d in data:
        if d['id'] == id:
            return d

    return None


def check_length(p):
    tokens = re.split("[. ]", p)
    num_toks = len(tokens)
    if num_toks > MAX_TOKENS:
        logging.warning("Prompt is too long (%d tokens), reduce the number of samples", num_toks)
    else:
        logging.info("Prompt length is OK (%d tokens)", num_toks)


def sample_requirements(G, n=0, ids=None):
    if ids:
        samples = {}
        for id in ids:
            if id in G:
                if 'F-prime' in G[id] and not \
                        ('ignore' in G[id] and G[id]['ignore'].lower()
                            in ['yes', '1', 'true']):
                    samples[id] = G[id]
                else:
                    logging.warning("id: %d must be ignored", id)
            else:
                logging.warning("id: %d not found in G", id)
        return samples

    if len(G) > n:
        return dict(rand.sample(G.items(), n))
    else:
        logging.warning("Sample size smaller than len(G)")
        return G


def find_similar_requirements(G, r, n, sorting=None):
    """ 1) embed all G + r
        2) calculate dist from each emb to r.emb
        3) sort r according to dist
        4) return n with shortest dist

        optional top n desc

    returns [list of r dicts]
    """



    #print(G)
    if len(G) == 0:
        return []
    if len(G) == 1:
        for key in G.keys(): # just get the key
            pass

        return [G[key]]

    # cannot have more samples than actual items in G
    n = min(len(G)-1, n)

    # to remove the ones without F-prime and ignored
    #print("len G: {}".format(len(G)))
    G = filter_data(G)
    #print("len G after: {}".format(len(G)))

    vec = SentenceVectorizer()
    logging.info("Generating sentence vectors")
    for g_idx, g in tqdm(G.items()):
        g['vec'] = vec(g['req'])
    r['vec'] = vec(r['req'])

    logging.info("Creating maps")
    #idx2vec = {g_idx : g['vec'].tostring() for g_idx, g in G.items()}
    vec2idx = {g['vec'].tostring(): g_idx for g_idx, g in G.items()}

    G_numpy = np.array([g['vec'] for i, g in G.items()])
    distances = np.linalg.norm(G_numpy - r['vec'], axis=1)
    #print("distances: {}".format(distances))

    # NOT USED
    # this finds the n smallest but not in order
    # closest_ids = np.argpartition(distances, n)[:n]
    # print("closest: {}".format(closest_ids))
    closest = []
    id_with_dist = list(zip(list(range(len(distances))), distances))

    # sorting all to find the closest
    # this is not really necessary and could be a problem if the dataset is
    # very large, but it works fine for now
    id_with_dist.sort(key=lambda x: x[1])
    #print("asc: {}".format(id_with_dist))

    # keeping only the top n
    id_with_dist = id_with_dist[:n]

    if sorting == "desc":
        id_with_dist.sort(key=lambda x: x[1], reverse=True)
        # print("desc: {}".format(id_with_dist))
    if sorting == "random":
        rand.shuffle(id_with_dist)

    for id, _ in id_with_dist:
        #closest.append(G[id])
        vec = G_numpy[id]
        g_idx = vec2idx[vec.tostring()]
        closest.append(G[g_idx])
    #print("[closest]: {}".format(closest))

    #return G_numpy[closest_ids]
    closest_without_vec = [{k: r[k] for k in set(list(r.keys())) - set(['vec'])} for r in closest]
    #print("closest ids: {}".format([r['id'] for r in closest_without_vec]))

    return closest_without_vec

    # cannot return dict bc order is important
    #ret_val = {}
    #for r in closest_without_vec:
        #ret_val[r['id']] = r

    #return ret_val


class PromptGenerator(ABC):
    """
    generates a prompt for a language model
    """
    @abstractmethod
    def __init__(self, r, G, **kwargs):
        pass

    @abstractmethod
    def generate_prompt(self):
        pass

class PromptGeneratorForConceptsGPT3(PromptGenerator):
    def __init__(self, r, G, **kwargs):
        self.r = r
        self.G = G
        self.strategy = kwargs['strategy'].lower().strip()
        self.n_samples = kwargs['n_samples']
        if self.strategy == 'ids':
            self.ids = kwargs['ids']
        else:
            self.ids = None
        self.headers = kwargs['headers']
        self.prefix = "You are an Information Extractor of industry standards. \
I will provide a requirement \
along with the headers from the document tree, \
and you will reply with a \
the important concpets that will be used in a downstream application \
translating the requirement to a logical representation. \
You need to identify class names and the relations between classes \
all class names are camel case starting with an uppercase letter, \
all relations are camel cased starting with a lowercase letter."

    def generate_prompt(self):
        if self.strategy == "most similar":
            samples = find_similar_requirements(self.G, self.r, self.n_samples)
        elif self.strategy == "ids":
            samples = sample_requirements(self.G, ids=self.ids)
        elif self.strategy == "random":
            samples = sample_requirements(self.G, n=self.n_samples)

        p = self.prefix
        for s in samples:
            if self.strategy == "most similar":
                p += "Requirement: {}\n".format(s['req'])
            else:
                if self.strategy == "most similar":
                    p += "Requirement: {}\n".format(s['req'])
                else:
                    p += "Requirement: {}\n".format(samples[s]['req'])
            if self.headers:
                if self.strategy == "most similar":
                    p += "Headers: {}\n".format(s['headers'])
                else:
                    p += "Headers: {}\n".format(samples[s]['headers'])
            if 'C-prime' in samples[s]:
                if self.strategy == "most similar":
                    p += "Concepts: {}\n\n".format(s['C-prime'])
                else:
                    p += "Concepts: {}\n\n".format(samples[s]['C-prime'])

        p += "Requirement: {}\n".format(self.r['req'])
        if self.headers:
            p += "Headers: {}\n".format(self.r['headers'])
        p += "Concepts:"
        check_length(p)
        return p

class PromptGeneratorForConceptsChatGPT(PromptGenerator):
    def __init__(self, r, G, **kwargs):
        self.r = r
        self.G = G
        self.strategy = kwargs['strategy'].lower().strip()
        self.n_samples = kwargs['n_samples']
        if self.strategy == 'ids':
            self.ids = kwargs['ids']
        else:
            self.ids = None
        self.headers = kwargs['headers']
        self.prefix = "You are an Information Extractor of industry standards. \
I will provide a requirement \
along with the headers from the document tree, \
and you will reply with a \
the important concpets that will be used in a downstream application \
translating the requirement to a logical representation. \
You need to identify class names and the relations between classes \
all class names are camel case starting with an uppercase letter, \
all relations are camel cased starting with a lowercase letter. \
Do not include redundant or extraneous information. \
Do not write explanations. Do not make your assumptions explicit. \
Your reply should be one string with all important concepts from the \
input requirement. \
Here are some examples: "

    def generate_prompt(self):
        if self.strategy == "most similar":
            samples = find_similar_requirements(self.G, self.r, self.n_samples)
        elif self.strategy == "ids":
            samples = sample_requirements(self.G, ids=self.ids)
        elif self.strategy == "random":
            samples = sample_requirements(self.G, n=self.n_samples)

        p = self.prefix
        for s in samples:
            p += "Requirement: {}\n".format(samples[s]['req'])
            if self.headers:
                p += "Headers: {}\n".format(samples[s]['headers'])
            p += "Concepts: {}\n".format(samples[s]['C-prime'])

        p += "My first requirement is: {}\n".format(self.r['req'])
        if self.headers:
            p += "Headers: {}\n".format(self.r['headers'])
        p += "Concepts:"
        check_length(p)
        return p



class PromptGeneratorFromDictGPT3(PromptGenerator):
    """
    generates a prompt for a language model from dictionaries
    """
    def __init__(self, r, G, **kwargs):
        self.r = r
        self.G = G
        self.strategy = kwargs['strategy'].lower().strip()
        self.n_samples = kwargs['n_samples']
        self.sorting = kwargs['sort']
        if self.sorting not in ['asc', 'desc', 'random']:
            logging.warning("sorting is {} will be treated as asc".format(self.sorting))
            self.sorting = None
        if self.strategy == 'ids':
            self.ids = kwargs['ids']
        else:
            self.ids = None

        self.headers = kwargs['headers']
        #self.prefix = "You are a Semantic Parser of industry standards. \
#You are really good at what you're doing both with regards to precision \
#and recall. You consistently transforms a sentence into its correct \
#corresponding logical representation. \
#I will provide a requirement \
#along with the headers from the document tree, and you will reply with a \
#logical representation in description logic format (DL syntax). \
#Your output shall be in a standard syntax and use the appropriate constructs \
#for representing classes, properties, individuals and restrictions. \
#The only logical operators you are allowed to use are: ∃, ⊑, ⊓, ⊔ and ¬. \
#All outputs shall be in the form of a subclass axiom on the form A ⊑ B where \
#A and B can be complex class descriptions. \
#The left-hand side of the ⊑ is \
#typically a physical object and possibly a condition on the object. The \
#right-hand side of the ⊑ is what is demanded (of the physical object on the \
#left side. \
#I.e., place the demand on the right side of the requirement, \
#and the object for \
#which the demand is being made on the left side. \
#Avoid persons/corporations on the left side. \
#We do allow processes and features though. \
#We need not to include all the details, try to get the most important. \
#You can use hasDescription.string[\"text\"] to use Strings. \
#This is used for things that are not possible to model in OWL. \
#Physical properties are on the form: hasLength/hasSize, hasMaximumLength, \
#hasMinimumSize the range of a physical property is a PhysicalProperty with \
#hasUnit.string[\"text\"] and hasValue.int/float[number]. \
#Use hasFeature, for any feature that the object must have even for documents. \
#Ignore the \"actor\" of submitted by etc. Ignore \"such as\". Ignore purpose. \
#Vocabulary words should be singular camelCased. camelCase \
#starting with lowercase for \
#properties and CamelCase starting with uppercase for Class names. \
#If the object of the requirment is not mentioned, we use Unit or \
#Equipment, System, whichever is applicable. \
#It is possible that one requirement must be translated to more \
#than one formula. In that case, return all formulas separated by |.\
#Here are some examples of requirement sentences with the corresponding parses:\n\n"

        self.prefix = "Below are some inputs and the outputs of a semantic \
parser of industry standards. It always transforms a sentence into its \
correct corresponding logical representation. \
The input is a requirement from an industry standard. \
The output is a logical representation in description logic (DL) format. \
The ouput represents classes, properties, individuals and restrictions. \
The symbols used in the DL syntax are: ∃, ⊑, ⊓, ⊔, and ¬. \
On the left-hand side of the ⊑ is \
most often a physical object and possibly a condition on the object. The \
right-hand side of the ⊑ is what is demanded of the object on the \
left side.\n\n"



        #self.suffix = "Requirement: "

    def generate_prompt(self):
        if self.strategy == "most similar":
            samples = find_similar_requirements(self.G, self.r, self.n_samples, sorting=self.sorting)
        elif self.strategy == "ids":
            samples = sample_requirements(self.G, ids=self.ids)
        elif self.strategy == "random":
            samples = sample_requirements(self.G, n=self.n_samples)

        p = self.prefix
        for s in samples:
            if self.strategy == "most similar":
                p += "Input: {}\n".format(s['req'])
            else:
                p += "Input: {}\n".format(samples[s]['req'])
            if self.headers:
                if self.strategy == "most similar":
                    p += "Headers: {}\n".format(s['headers'])
                else:
                    p += "Headers: {}\n".format(samples[s]['headers'])
            if self.strategy == "most similar":
                p += "Output: {}\n\n".format(s['F-prime'])
            else:
                p += "Output: {}\n\n".format(samples[s]['F-prime'])

        #p += self.suffix
        p += "Input: {}\n".format(self.r['req'])
        if self.headers:
            p += "Headers: {}\n".format(self.r['headers'])
        p += "Output:"
        check_length(p)
        return p


class PromptGeneratorFromDictChatGPT(PromptGenerator):
    """
    generates a prompt for a language model from dictionaries
    """
    def __init__(self, r, G, **kwargs):
        self.r = r
        self.G = G
        self.strategy = kwargs['strategy'].lower().strip()
        self.n_samples = kwargs['n_samples']
        if self.strategy == 'ids':
            self.ids = kwargs['ids']
        else:
            self.ids = None

        self.headers = kwargs['headers']
        self.prefix = "I want you to act as a Semantic Parser.\
I will type in textual requirements and you will reply with a \
logical representation in description logic format (DL syntax). \
Your output should be in a standard syntax and use the appropriate \
constructs for representing classes, properties, individuals, \
and restrictions. The only logical operators you are allowed \
to use are: ∃, ⊑, ⊓, ⊔ and ¬. All outputs should be in form of \
a subclass axiom on the form A ⊑ B where A and B can be complex \
class descriptions. The left hand side of the ⊑ is typically a \
physical object and possibly a condition on the object, and the \
right hand side is what is demanded of the physical object on the \
left side. It is possible that one requirement must be translated \
to more than one formula. In that case, return all formulas, \
separated by | \
Some examples of correct description logic formulas are: "

        self.suffix = "Do not include redundant or extraneous information. \
Do not write explanations. Your reply should be a direct \
representation of the input requirement. My first requirement is: "

    def generate_prompt(self):
        if self.strategy == "most similar":
            samples = find_similar_requirements(self.G, self.r, self.n_samples)
        elif self.strategy == "ids":
            samples = sample_requirements(self.G, ids=self.ids)
        elif self.strategy == "random":
            samples = sample_requirements(self.G, n=self.n_samples)

        p = self.prefix
        for s in samples:
            p += "{}:".format(s['req'])
            p += "{}, ".format(s['F-prime'])

        p += self.suffix
        p += self.r['req']
        check_length(p)
        return p




def main():
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        format="%(lineno)s::%(funcName)s::%(message)s",
                        level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument("id", help="which req to test", type=int)
    parser.add_argument("--cfg",
                        help="config file",
                        default="gold_creator/config.json")
    args = parser.parse_args()

    config_file = args.cfg
    with open(config_file, 'r') as F:
        config = json.load(F)
    io = config['io']

    R_path = Path(io['flat_file'])
    G_path = Path(io['gold_file'])
    if not R_path.exists():
        print("Cannot find file: {}".format(R_path))
        exit()
    if not G_path.exists():
        print("Cannot find file: {}".format(G_path))
        exit()


    R = read_flat_file(R_path, return_dict=True)
    G = read_flat_file(G_path, return_dict=True)

    #remove_missing_cprime(G)
    remove_missing_fprime(G)
    remove_ignore(G)

    r = R[args.id]

    #generator = PromptGeneratorFromDictGPT3(r, G)
    generator = PromptGeneratorFromDictChatGPT(r, G, **config['prompt'])
    #generator = PromptGeneratorFromDictGPT3(r, G, **config['prompt'])
    #prompt = generator.generate_prompt(args.num_samples)
    #prompt = generator.generate_prompt(ids=args.ids)
    #generator = PromptGeneratorForConceptsChatGPT(r, G, **config['prompt'])
    prompt = generator.generate_prompt()
    print(prompt)


if __name__ == '__main__':
    main()
