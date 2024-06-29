import logging
import re
import nltk
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from utils.io import read_flat_file, write_flat_file_in_order, remove_ignore, remove_missing_fprime
from utils.vectorizer import SentenceVectorizer
from jiwer import wer
#from torchmetrics import TranslationEditRate
from sacrebleu.metrics.lib_ter import translation_edit_rate
from sacrebleu.metrics import CHRF
import matplotlib.pyplot as plt

vec = SentenceVectorizer()

def process_stack_old(g, stack):
    while stack:
        e, p = stack.pop()
        g.add_node(e)
        g.add_edge(e, p)
        parts = split_formula(e)
        if len(parts) == 1:
            parts = split_formula_on_symbol(e, '.')
        if len(parts) > 1:
            for part in parts:
                if part[0] == '(' and part[-1] == ')':
                    part = part[1:-1]
                stack.append((part,e))


def remove_parenthesis(e):
    if e and e[0] == '(' and e[-1] == ')':
        e = e[1:-1]
    return e


def process_stack(g, stack):
    """
    The graph is:
        - without edge labels
        - connector symbols at each node
        - class names at the leaves
        - for restrictions, we keep ∃r as the node name
    """
    last_node_id = -1
    axiom_edge_id = 0
    while stack:
        sub_formula, parent_id = stack.pop()
        axiom = split_formula_on_symbol(sub_formula, '⊑')

        if len(axiom) == 2:
            # handle axiom
            id = last_node_id + 1
            g.add_node(id, label='⊑')
            if id > 0:  # a badly formlated axiom
                if parent_id == 0:
                    g.add_edge(id, parent_id, label=axiom_edge_id)
                    axiom_edge_id += 1
                else:
                    g.add_edge(id, parent_id)
            left, right = axiom
            stack.append((left, id))
            stack.append((right, id))
            last_node_id = id

        elif len(axiom) > 2:
            # this is really a syntactic error:
            # handle syntactic error:
            # just parse from left to right as we do with restrictions
            left = axiom[0]
            right = '⊑'.join(axiom[1:])
            id = last_node_id + 1
            g.add_node(id, label='⊑')
            stack.append((left, id))
            stack.append((right, id))
            last_node_id = id

        else:
            conjuncts = split_formula_on_symbol(sub_formula, '⊓')

            if len(conjuncts) > 1:
                # handle conjuncts
                id = last_node_id + 1
                g.add_node(id, label="⊓")
                if parent_id == 0:
                    g.add_edge(id, parent_id, label=axiom_edge_id)
                    axiom_edge_id += 1
                else:
                    g.add_edge(id, parent_id)
                for conjunct in conjuncts:
                    conjunct = remove_parenthesis(conjunct)
                    stack.append((conjunct, id))
                last_node_id = id

            else:
                disjuncts = split_formula_on_symbol(sub_formula, '⊔')

                if len(disjuncts) > 1:
                    # handle disjuncts
                    id = last_node_id + 1
                    g.add_node(id, label="⊔")
                    if parent_id == 0:
                        g.add_edge(id, parent_id, label=axiom_edge_id)
                        axiom_edge_id += 1
                    else:
                        g.add_edge(id, parent_id)
                    for disjunct in disjuncts:
                        disjunct = remove_parenthesis(disjunct)
                        stack.append((disjunct, id))
                    last_node_id = id

                else:
                    restriction = split_formula_on_symbol(sub_formula, '.')
                    #print("restriction: {}".format(restriction))
                    #print("len(restriction): {}".format(len(restriction)))

                    if len(restriction) == 1:
                        # a leaf
                        id = last_node_id + 1
                        #g.add_node(sub_formula)
                        g.add_node(id, label=sub_formula)
                        if parent_id == 0:
                            g.add_edge(id, parent_id, label=axiom_edge_id)
                            axiom_edge_id += 1
                        else:
                            g.add_edge(id, parent_id)
                        last_node_id = id

                    elif len(restriction) == 2:
                        property, concept = restriction
                        concept = remove_parenthesis(concept)
                        #print("property: {}".format(property))
                        #print("concept: {}".format(concept))
                        #input()
                        id = last_node_id + 1
                        g.add_node(id, label=property)
                        if parent_id == 0:
                            g.add_edge(id, parent_id, label=axiom_edge_id)
                            axiom_edge_id += 1
                        else:
                            g.add_edge(id, parent_id)
                        stack.append((concept, id))
                        last_node_id = id

                    elif len(restriction) > 2:
                        # either a "combined" restriction or a grammar mistrake
                        # we add the first level, and push the rest back onto the
                        # stack
                        property = restriction[0]
                        concept = ".".join(restriction[1:])
                        id = last_node_id + 1
                        g.add_node(id, label=property)
                        if parent_id == 0:
                            g.add_edge(id, parent_id, label=axiom_edge_id)
                            axiom_edge_id += 1
                        else:
                            g.add_edge(id, parent_id)
                        stack.append((concept, id))
                        last_node_id = id

                    else:
                        logging.warning("Something wrong? restriction: {}".format(restriction))

    return last_node_id


def parse_dl(s):
    """
    parses a dl expression into a graph
    """
    g = nx.Graph()

    stack = [(s, 0)]
    num_nodes = process_stack(g, stack)




    #pos = nx.spring_layout(g)
    #pos = graphviz_layout(g, prog="twopi")
    #pos = graphviz_layout(g, prog="dot")
    #print(g.nodes.data())
    #nx.draw(g,
            #pos,
            #with_labels=True,
            #labels={key : value['label'] for key, value in dict(g.nodes(data=True)).items()})
#
    #nx.draw_networkx_edge_labels(g,pos,font_color='red')
    #nx.draw_networkx_edge_labels(g,
                                 #pos,
                                 #font_color='red')
                                 #labels={(e1,e2) : d['label'] for (e1,e2), d  in dict(g.edges(data=True)).items()})

    #plt.show()
    #print(s)
    #input()


    return g


def nsub_cost(g1, g2):
    if g1['label'] != g2['label']:
        #print("sub: {} and {}".format(g1, g2))
        return 1
    else:
        return 0

def ndel_cost(node):
    return 1

def nins_cost(node):
    return 1

def esub_cost(e1, e2):
    # Return 1 if the edges are unequal:
    # Either both have labels and they are different
    # Or one have label and the other has no label
    if 'label' in e1:
        if 'label' in e2 and e1['label'] == e2['label']:
            return 0
        else:
            return 1
    elif 'label' in e2:
        return 1
    else:
        return 0

def edel_cost(none):
    return 1

def eins_cost(none):
    return 1

def graph_edit_distance(s1, s2):
    g1 = parse_dl(s1)
    g2 = parse_dl(s2)

    #print("calculating graph edit distance")
    empty_g = nx.Graph()
    distance = nx.graph_edit_distance(g1, g2,
                                      #upper_bound=20,
                                      roots=(0, 0),
                                      timeout=20,
                                      node_subst_cost=nsub_cost,
                                      node_del_cost=ndel_cost,
                                      node_ins_cost=nins_cost,
                                      edge_subst_cost=esub_cost,
                                      edge_del_cost=edel_cost,
                                      edge_ins_cost=eins_cost)

    #print("distance: {}".format(distance))
    empty_g.add_node(0, label='⊑')
    empty_distance = nx.graph_edit_distance(empty_g, g2,
                                      #upper_bound=20,
                                      roots=(0, 0),
                                      timeout=20,
                                      node_subst_cost=nsub_cost,
                                      node_del_cost=ndel_cost,
                                      node_ins_cost=nins_cost)
    #print("empty_distance: {}".format(empty_distance))

    if distance is None:
        distance = 9999
    if empty_distance is None:
        empty_distance = 9999

    min_distance = min(distance, empty_distance)
    #distances = nx.optimize_graph_edit_distance(g1, g2)
    #for d in distances:
    #    print("graph edit distance: {}".format(d))

    #print("graph edit distance: {}".format(distance))
    return int(min_distance)



def split_formula_on_symbol(f, symbol):
    result = []
    balance = 0
    start = 0
    for i, char in enumerate(f):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif char == symbol and balance == 0:
            result.append(f[start:i].strip())
            start = i + 1
    result.append(f[start:].strip())
    return result


def split_formula(f):
    result = []
    balance = 0
    start = 0
    for i, char in enumerate(f):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif (char == '⊓' or char == '⊔' or char == '⊑') and balance == 0:
            result.append(f[start:i].strip())
            start = i + 1
    result.append(f[start:].strip())
    return set(result)


def changed_parts(s1, s2):
    """
    s1: f
    s2: f-prime
    returns 1 - number of unchanged elements on the correct side / number of elements in f-prime
    """
    s1_ax = re.split(r'⊒', s1)
    s2_ax = re.split(r'⊒', s2)
    s1_left = s1_ax[0]
    s1_right = s1_ax[1]
    s2_left = s2_ax[0]
    s2_right = s2_ax[1]

    s1_left_parts = split_formula(s1_left)
    s1_right_parts = split_formula(s1_right)
    s2_left_parts = split_formula(s2_left)
    s2_right_parts = split_formula(s2_right)


    left_intersection = s1_left_parts.intersection(s2_left_parts)
    right_intersection = s1_right_parts.intersection(s2_right_parts)
    len_left = len(left_intersection)
    len_right = len(right_intersection)

    num_unchanged = len_left + len_right
    num_elements = len(s2_left_parts) + len(s2_right_parts)
    part_unchanged = num_unchanged / num_elements

    return 1 - part_unchanged


def cosine_similarity(array1, array2):
    dot_product = np.dot(array1, array2)
    magnitude1 = np.sqrt(np.dot(array1, array1))
    magnitude2 = np.sqrt(np.dot(array2, array2))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity


def semantic_distance(s1, s2):
    s1_vec = vec.embed(s1)
    s2_vec = vec.embed(s2)
    simil = cosine_similarity(s1_vec, s2_vec)
    dist = 1 - simil
    return dist

def formula2tok(s):
    """
    split the formula into single tokens
    """
    s = re.sub("\\[", " ", s) # just remove square brackets for string/float
    s = re.sub("\\]", "", s)
    s = re.sub("∃", "∃ ", s)
    s = re.sub("∀", "∀ ", s)
    s = re.sub("(≥\\d)", "\\1 ", s)
    s = re.sub("(≤\\d)", "\\1 ", s)
    s = re.sub("(=\\d)", "\\1 ", s)
    s = re.sub("¬", "¬ ", s)
    s = re.sub("[\\(\\)]", "", s)
    tokens = re.split("[ .]", s)
    return tokens

def formula2set(s):
    """
    convert formula to set of tokens
    """
    return set(formula2tok(s))


def jaccard_distance(f1, f2):
    set1 = formula2set(f1)
    set2 = formula2set(f2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_index = len(intersection) / len(union)
    jaccard_distance = 1 - jaccard_index
    return jaccard_distance

def edit_distance(f1, f2):
    l1 = formula2tok(f1)
    l2 = formula2tok(f2)
    l_empty = formula2tok("")
    #print("tokens1: {}".format(l1))
    #print("tokens2: {}".format(l2))
    score = nltk.edit_distance(l1, l2, transpositions=True)
    score_empty = nltk.edit_distance(l_empty, l2, transpositions=True)
    #print("edit distance: {}".format(score))
    #input()
    return min(score, score_empty)

def translation_error_rate(pred, gold):
    tokenized_pred = formula2tok(pred)
    tokenized_gold = formula2tok(gold)
    #print("tokens1: {}".format(tokenized_pred))
    #print("tokens2: {}".format(tokenized_gold))

    edit, gold_length = translation_edit_rate(tokenized_pred, tokenized_gold)
    score = (edit / gold_length) if gold_length > 0 else 1
    #print(score)
    #input()
    return min(1, score)

def chrf_score(pred, gold):
    chrf = CHRF()
    score = chrf.corpus_score(pred, [gold])
    #print("pred: {}".format(pred))
    #print("gold: {}".format(gold))
    #print(score)
    #input()
    return score.score



def evaluate_one(g, **kwargs):
    results = {}
    f = g['F']
    fprime = g['F-prime']

    if kwargs['normalize']:
        if kwargs['edit']:
            if 'edit' in g:
                edit = int(g['edit'])
            else:
                edit = edit_distance(f, fprime)
            l = formula2tok(fprime)
            norm_edit = edit / len(l)
            #print("norm_edit: {}".format(norm_edit))
            results['norm_edit'] = norm_edit
            #input()
        if kwargs['graph_edit']:
            if 'graph_edit' in g:
                gedit = int(g['graph_edit'])
            else:
                gedit = graph_edit_distance(f, fprime)
            g = parse_dl(fprime)
            num_nodes = g.number_of_nodes()
            num_edges = g.number_of_edges()
            norm_gedit = gedit / (num_nodes + num_edges)
            #print("norm_graph_edit: {}".format(norm_gedit))
            results['norm_graph_edit'] = norm_gedit
            #input()

    else:
        if kwargs['jaccard']:
            jaccard_score = jaccard_distance(f, fprime)
            results['jaccard'] = jaccard_score
        if kwargs['wer']:
            wer_score = wer(fprime, f)
            if kwargs['wer_max']:
                results['wer'] = min(wer_score, 1)
            else:
                results['wer'] = wer_score
        if kwargs['semantic']:
            semantic_score = semantic_distance(f, fprime)
            results['semantic'] = semantic_score
        if kwargs['parts']:
            parts_score = changed_parts(f, fprime)
            results['parts'] = parts_score
        if kwargs['edit']:
            edit = edit_distance(f, fprime)
            results['edit'] = edit
        if kwargs['graph_edit']:
            graph_edit = graph_edit_distance(f, fprime)
            results['graph_edit'] = graph_edit
        if kwargs['ter']:
            edit = translation_error_rate(f, fprime)
            results['ter'] = edit
        if kwargs['chrf']:
            edit = chrf_score(f, fprime)
            results['chrf'] = edit

    return results


def evaluate_all(G, **kwargs):
    logging.info("Calculating measures")
    for key, g in tqdm(G.items()):
        if 'ignore' in g  \
                and g['ignore'].lower() not in ['false', 'no', '0']:
            continue
        # cannot calculate if this is not present
        elif 'F-prime' not in g or 'F' not in g:
            print("F not found")
            print(g)
            input()
            continue
        elif not g['F-prime']:
            print("F-prime is empty")
            continue
        else:
            score_dict = evaluate_one(g, **kwargs)
            if kwargs['normalize']:
                if kwargs['edit']:
                    g['norm_edit'] = score_dict['norm_edit']
                if kwargs['graph_edit']:
                    g['norm_graph_edit'] = score_dict['norm_graph_edit']
            else:
                if kwargs['jaccard']:
                    g['jaccard'] = score_dict['jaccard']
                if kwargs['wer']:
                    g['wer'] = score_dict['wer']
                if kwargs['semantic']:
                    g['semantic'] = score_dict['semantic']
                if kwargs['parts']:
                    g['parts'] = score_dict['parts']
                if kwargs['edit']:
                    g['edit'] = score_dict['edit']
                if kwargs['graph_edit']:
                    g['graph_edit'] = score_dict['graph_edit']
                if kwargs['ter']:
                    g['ter'] = score_dict['ter']
                if kwargs['chrf']:
                    g['chrf'] = score_dict['chrf']


def main():
    parser = ArgumentParser()
    parser.add_argument("input", help='gold file to be evaluated')
    parser.add_argument("--semantic", "-s", help="use semantic measure",
                        action="store_true")
    parser.add_argument("--wer", "-w", help="use wer",
                        action="store_true")
    parser.add_argument("--jaccard", "-j", help="use jaccard distance",
                        action="store_true")
    parser.add_argument("--parts", "-p", help="changed parts",
                        action="store_true")
    parser.add_argument("--wer_max", help="max wer as 1",
                        action="store_true")
    parser.add_argument("--edit", "-e", help="use edit distance",
                        action="store_true")
    parser.add_argument("--graph_edit", "-g", help="graph edit distance",
                        action="store_true")
    parser.add_argument("--ter", "-t", help="Translation Error Rate",
                        action="store_true")
    parser.add_argument("--chrf", "-c", help="chrf", action="store_true")
    parser.add_argument("--id", help="calculate and display only one id",
                        type=int)
    parser.add_argument("--normalize",
                        "-n",
                        help="create a normalized version of the score (edit/graph_edit), this will reuse the existing value and add another row e.g., norm_edit",
                        action="store_true")

    args = parser.parse_args()
    #print(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Cannot find: {}, exiting...".format(input_path))
        exit()

    G = read_flat_file(input_path, return_dict=True)
    #if args.id:
    #    if args.id in G:
    #        print("text: {}".format(G[args.id]['req']))
    #        print("F: {}".format(G[args.id]['F']))
    #        print("F-prime: {}".format(G[args.id]['F-prime']))
    #        score_dict = evaluate_one(G[args.id], **vars(args))
    #        print(score_dict)
#
#    else:
    evaluate_all(G, **vars(args))
    write_flat_file_in_order(input_path, G)


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()], format="%(lineno)s::%(funcName)s::%(message)s", level=logging.DEBUG)
    main()
