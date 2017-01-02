import random

import numpy as np

from data_simulation import generate_case, rate_sub_HKY, scale_branches_length
from felsenstein import pruning
from tree_serialisation import load_tree
from viterbi_sumproduct import sum_product, viterbi


def single_decoding_routine(tree_path, number_of_nucleotids, alphabet, A,
                            b, n_species, pi, kappa, alg='viterbi'):
    '''Generates a sequence of states and an associated list of strands. Then
    decodes those using a phylogenetic HMM model.
    '''
    nbState = A.shape[0]
    # load the phylogenetic model from JSON
    tree = load_tree(tree_path)
    trees = []

    for j in range(nbState):
        scaling_factor = random.random()
        trees.append(scale_branches_length(tree, scale=scaling_factor))

    strands, states = generate_case(A, b, pi, kappa,
                                    trees, number_of_nucleotids)
    # Transform strands from ints to strings
    str_strands = list()
    for strand in strands:
        str_strand = ""
        for acid_int in strand:
            str_strand = ''.join([str_strand, alphabet[acid_int]])
        str_strands += [str_strand]
        # Transform strands in sites
    sites = list()
    for site_ind in range(number_of_nucleotids):
        sites += [''.join([str_strands[species_ind][site_ind]
                           for species_ind in range(n_species)])]
    # Process likelihoods with Felsenstein's algorithm
    Qs = rate_sub_HKY(pi, kappa)
    likelihoods = np.zeros((nbState, number_of_nucleotids))
    for state in range(nbState):
        tree = trees[state]
        Q = Qs[state]
        p = pi[state]
        for site_ind, site in enumerate(sites):
            likelihoods[state, site_ind] = pruning(Q, p, tree, site)
    # VITERBI PARAMETERS
    S = range(nbState)
    if alg == 'viterbi':
        state_sequence_decoded = viterbi(S, A, b, likelihoods)
    elif alg == 'sp':
        state_sequence_decoded = np.argmax(sum_product(S, A, b, likelihoods),
                                           axis=0)
    # Metrics
    precision = np.sum(states == state_sequence_decoded) / number_of_nucleotids
    reference_codon_sequence = translate_states_to_codons(states)
    decoded_codon_sequence = translate_states_to_codons(state_sequence_decoded)
    sens = sensitivity(reference_codon_sequence, decoded_codon_sequence)
    spec = specificity(reference_codon_sequence, decoded_codon_sequence)
    return precision, sens, spec


def translate_states_to_codons(states):
    '''Transforms a state sequence into a codon/non-codon sequence. With states
    corresponding to codons being (0-1-2) et non-codons being (3).
        Args :
            - states (list or ndarray): the sequence of state
        Output:
            - ndarray: the corresponding codon(1)/non-codon(0) sequence
    '''
    codon_sequence = list()
    n_nucleotids = len(states)
    for state_ind, state in enumerate(states):
        if int(state) == 3:
            codon_sequence += [0]
        else:
            codon_sequence += [1]
    return np.array(codon_sequence)


def sensitivity(reference_codon_sequence, comparison_codon_sequence):
    '''Calculates the true positive rate
        Args:
            - reference_codon_sequence (list or ndarray): the true codon
            sequence
            - comparison_codon_sequence (list or ndarray): the codon to compare
            to the reference sequence
        Output:
            - float: the TP rate
    '''
    positive_predictions = comparison_codon_sequence == 1
    true_predicitons = np.equal(
        reference_codon_sequence, comparison_codon_sequence)
    true_positives = positive_predictions * true_predicitons
    return np.sum(true_positives) / np.sum(positive_predictions)


def specificity(reference_codon_sequence, comparison_codon_sequence):
    '''Calculates the true negative rate
        Args:
            - reference_codon_sequence (list or ndarray): the true codon
            sequence
            - comparison_codon_sequence (list or ndarray): the codon to compare
            to the reference sequence
        Output:
            - float: the TN rate
    '''
    negative_predictions = comparison_codon_sequence == 0
    true_predicitons = np.equal(
        reference_codon_sequence, comparison_codon_sequence)
    true_negatives = negative_predictions * true_predicitons
    return np.sum(true_negatives) / np.sum(negative_predictions)
