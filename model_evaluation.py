import random
from copy import deepcopy

import numpy as np

from data_simulation import generate_case, rate_sub_HKY, scale_branches_length
from felsenstein import pruning, np_pruning
from tree_serialisation import load_tree
from viterbi_sumproduct import sum_product, viterbi


def gene_finding_metrics(real_states, decoded_states):
    '''Computes the three metrics we need in the first example.
    '''
    precision = np.sum(real_states == decoded_states) / number_of_nucleotids
    reference_codon_sequence = translate_states_to_codons(real_states)
    decoded_codon_sequence = translate_states_to_codons(decoded_states)
    sens = sensitivity(reference_codon_sequence, decoded_codon_sequence)
    spec = specificity(reference_codon_sequence, decoded_codon_sequence)
    return precision, sens, spec


def conserved_regions_metrics(real_states, posterior_proba_interest,
                              state_of_interest=0, steps=0.1, thresholds=None):
    '''Computes a false positive rate and true positive rate for different
    threshold values
    '''
    if thresholds is None:
        thresholds = np.arange(0, 1, steps)
    interest_values = real_states == state_of_interest
    fp_rates = np.zeros(len(thresholds))
    tp_rates = np.zeros(len(thresholds))
    for t_idx, threshold in enumerate(thresholds):
        identified_interest = posterior_proba_interest > threshold
        true_predicitons = np.equal(interest_values, identified_interest)
        false_predictions = np.not_equal(interest_values, identified_interest)
        correctly_identified_p = identified_interest * true_predicitons
        tp_rates[t_idx] = np.sum(correctly_identified_p) / np.sum(
            interest_values)
        wrongly_identified_p = identified_interest * false_predictions
        fp_rates[t_idx] = np.sum(wrongly_identified_p) / np.sum(
            1-interest_values)
    return tp_rates, fp_rates


def single_decoding_routine(tree_path, number_of_nucleotids, alphabet, A,
                            b, n_species, pi, kappa, scaling_factors,
                            alg='viterbi', list_of_species=[]):
    '''Generates a sequence of states and an associated list of strands. Then
    decodes those using a phylogenetic HMM model.
    '''

    labs = []
    times = []
    nbState = A.shape[0]

    # load the phylogenetic model from JSON
    tree = load_tree(tree_path)

    trees = []

    if list_of_species:
        tree = sub_tree(tree, list_of_species)
        n_species = len(list_of_species)

    for j in range(nbState):
        trees.append(scale_branches_length(tree, scale=scaling_factors[j]))

    strands, states = generate_case(A, b, pi, kappa,
                                    trees, number_of_nucleotids)

    # Process likelihoods with Felsenstein's algorithm

    Qs = rate_sub_HKY(pi, kappa)
    # Process likelihoods with Felsenstein's algorithm

    likelihoods = np.zeros((nbState, number_of_nucleotids))
    sites = np.zeros((number_of_nucleotids, n_species))
    for i in range(n_species):
        sites[:, i] = strands[i]
    for state in range(nbState):
        tree = trees[state]
        Q = Qs[state]
        p = pi[state]
        likelihoods[state] = np_pruning(Q, p, tree, sites)

    # VITERBI PARAMETERS
    S = range(nbState)

    if alg == 'viterbi':
        return {"real_states": states,
                "decoded_states": viterbi(S, A, b, likelihoods)}
    elif alg == 'sp':
        probabilities = sum_product(A, b, likelihoods)
        return {"real_states": states,
                "probabilities": probabilities,
                "decoded_states": np.argmax(probabilities, axis=0)}


def translate_states_to_codons(states):
    '''Transforms a state sequence into a codon/non-codon sequence. With states
    corresponding to codons being (0-1-2) et non-codons being (3).
        Args :
            - states (ndarray): the sequence of state
        Output:
            - ndarray: the corresponding codon(1)/non-codon(0) sequence
    '''
    return 1 - (states == 3)


def sensitivity(reference_codon_sequence, comparison_codon_sequence):
    '''Portion correctly predicted of sites actually in genes
        Args:
            - reference_codon_sequence (ndarray): the true codon
            sequence
            - comparison_codon_sequence (ndarray): the codon to compare
            to the reference sequence
        Output:
            - float: the sensitivity
    '''
    positive_predictions = comparison_codon_sequence == 1
    true_predicitons = np.equal(
        reference_codon_sequence, comparison_codon_sequence)
    true_positives = positive_predictions * true_predicitons
    return np.sum(true_positives) / np.sum(true_predicitons)


def specificity(reference_codon_sequence, comparison_codon_sequence):
    '''Portion correct of sites predicted to be in genes
        Args:
            - reference_codon_sequence (ndarray): the true codon
            sequence
            - comparison_codon_sequence (ndarray): the codon to compare
            to the reference sequence
        Output:
            - float: the specificity
    '''
    positive_predictions = comparison_codon_sequence == 1
    true_predicitons = np.equal(
        reference_codon_sequence, comparison_codon_sequence)
    true_positives = positive_predictions * true_predicitons
    if np.sum(positive_predictions) == 0:
        import pdb
        pdb.Pdb().set_trace()
    return np.sum(true_positives) / np.sum(positive_predictions)


def sub_tree(tree, list_of_species):
    """
    Prune the tree to keep only keep species in a list
    Args :
           - tree (dict)
           - list_of_species (int list)list of index
    """
    tree_cp = deepcopy(tree)

    def treat_node(node):
        children = tree[node]  # important for the recursion
        if children:  # propagate recirsively
            for child in children:
                treat_node(child["node"])

            for child in children:
                new_child = child["node"]
                if new_child not in list_of_species and not tree_cp[new_child]:
                    # we drop the child
                    tree_cp[node].remove(child)
                    # discard "dead leaf"
                    tree_cp.pop(new_child, None)

    treat_node(max(tree_cp.keys()))

    # now merge useless intermediary node ie with only one son
    def merge_unary(node, ancestor_list):
        if(len(tree_cp[node]) == 1):
            child = tree_cp[node][0]["node"]
            # if the child has ancestor we can remove _node_
            if ancestor_list:
                branch_length = tree_cp[node][0]["branch"]
                parent = ancestor_list[0]
                siblings = tree_cp[parent]
                for sibling in siblings:
                    if sibling['node'] == node:
                        sibling['node'] = child
                        sibling['branch'] += branch_length
                # child ancestor are now exactly node's ancestor
                merge_unary(child, ancestor_list)
            # if the child has no grand parent but has children
            elif tree_cp[tree_cp[node][0]["node"]]:
                tree_cp[child][0]["branch"] += tree_cp[node][0]["branch"]
                merge_unary(child, [])
            tree_cp.pop(node, None)
        elif len(tree_cp[node]) == 2:
            merge_unary(tree_cp[node][0]["node"], [node] + ancestor_list)
            merge_unary(tree_cp[node][1]["node"], [node] + ancestor_list)

    merge_unary(max(tree_cp.keys()), [])
    tree_rn = {}

    def rename(node):
        # rename nodes so that all indices are betwen 1 2*n -1
        children = tree_cp[node]
        for child in children:
            child_node = child["node"]
            child["node"] = sorted(tree_cp.keys()).index(child_node)+1
            rename(child_node)  # rename child first
        # now rename actual node
        node_renamed = sorted(tree_cp.keys()).index(node)+1
        tree_rn[node_renamed] = tree_cp[node]

    rename(max(tree_cp.keys()))
    return tree_rn
