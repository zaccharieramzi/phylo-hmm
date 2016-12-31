import numpy as np


def translate_states_to_codons(states):
    '''Transforms a state sequence into a codon/non-codon sequence. With states
    corresponding to codons being (0-1-2) et non-codons being (3).
        Args :
            - states (list or ndarray): the sequence of state
        Output:
            - ndarray: the corresponding codon(1)/non-codon(0) sequence
        Errors:
            - ValueError: when the state sequence contains impossible
            subsequences.
    '''
    codon_sequence = list()
    n_nucleotids = len(states)
    for state_ind, state in enumerate(states):
        if int(state) == 3:
            codon_sequence += [0]
        if int(state) == 0:
            if state_ind == n_nucleotids-1:
                codon_sequence += [1]
            elif state_ind == n_nucleotids-2:
                if int(states[-1]) == 1:
                    codon_sequence += [1]
                else:
                    raise ValueError("A 0 state is not followed by 1")
            else:
                if (int(states[state_ind+1]) == 1) and\
                  (int(states[state_ind+2]) == 2):
                    codon_sequence += [1]
                else:
                    raise ValueError("A 0 state is not followed by 1-2")
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
    true_predicitons = reference_codon_sequence == comparison_codon_sequence
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
    true_predicitons = reference_codon_sequence == comparison_codon_sequence
    true_negatives = negative_predictions * true_predicitons
    return np.sum(true_negatives) / np.sum(negative_predictions)