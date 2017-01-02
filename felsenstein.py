import numpy as np
from scipy.linalg import expm


def pruning(Q, pi, tree, site):
    '''Computes the likelihood at a given site, for a given instance of this
    site with respect to a phylogenetic tree defined by Q, pi, tree, using
    Felsenstein's pruning algorithm.
        Args:
            - Q (np.matrix): the subsitution rate matrix.
            - pi (np.array): the vector of background frequencies.
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - site (list): the instance of the site.
        Output:
            - float: the likelihood.
    '''
    dynamic_probas = {}  # this dictionary will allow to keep records of
    # posterior probabilities which were already calculated.
    nucleotide_map = {
        "A": 0,
        "C": 1,
        "T": 2,
        "G": 3
    }
    n = len(site)
    observation = np.zeros(n)
    for i, nucleotide in enumerate(site):
        observation[i] = nucleotide_map[nucleotide]

    def posterior_proba(node):
        if node in dynamic_probas:
            return dynamic_probas[node]
        else:
            childs = tree[node]
            if childs:
                left_child = childs[0]["node"]
                right_child = childs[1]["node"]
                post_proba_left = posterior_proba(left_child)
                post_proba_right = posterior_proba(right_child)
                prob_matrix_left = expm(childs[0]["branch"]*Q)
                prob_matrix_right = expm(childs[1]["branch"]*Q)
                left_likelihood = prob_matrix_left.dot(post_proba_left)
                right_likelihood = prob_matrix_right.dot(post_proba_right)
                return left_likelihood*right_likelihood
            else:
                # in this case the node is a leaf
                likelihood = np.zeros(4)
                likelihood[int(observation[node - 1])] = 1
                return likelihood

    for node in tree:
        dynamic_probas[node] = posterior_proba(node)
    return dynamic_probas[2*n-1].dot(pi)


def np_pruning(Q, pi, tree, sites):
    '''Computes the likelihood at a given site, for a given instance of this
    site with respect to a phylogenetic tree defined by Q, pi, tree, using
    Felsenstein's pruning algorithm.
        Args:
            - Q (np.matrix): the subsitution rate matrix.
            - pi (np.array): the vector of background frequencies.
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - site (np matrix): all the sites
        Output:
            - (np vector): the likelihood.
    '''
    dynamic_probas = {}  # this dictionary will allow to keep records of
    # posterior probabilities which were already calculated.

    nb_nucleotides, n_species = sites.shape
    observation = sites.copy()
    x_index = np.arange(nb_nucleotides)  # frequently used variable

    def posterior_proba(node):
        if node in dynamic_probas:
            return dynamic_probas[node]
        else:
            childs = tree[node]
            if childs:
                left_child = childs[0]["node"]
                right_child = childs[1]["node"]
                post_proba_left = posterior_proba(left_child)
                post_proba_right = posterior_proba(right_child)
                prob_matrix_left = expm(childs[0]["branch"]*Q)
                prob_matrix_right = expm(childs[1]["branch"]*Q)
                # import pdb; pdb.set_trace()
                left_likelihood = prob_matrix_left.dot(post_proba_left.T)
                right_likelihood = prob_matrix_right.dot(post_proba_right.T)
                return (left_likelihood*right_likelihood).T
            else:
                # in this case the node is a leaf
                likelihood = np.zeros((nb_nucleotides, 4))
                likelihood[x_index, np.floor(
                    observation[:, node - 1]).astype('int')] = 1
                return likelihood

    for node in tree:
        dynamic_probas[node] = posterior_proba(node)

    return dynamic_probas[max(tree.keys())].dot(pi)
