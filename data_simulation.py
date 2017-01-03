import numpy as np
from scipy.linalg import expm
from copy import deepcopy
import argparse

from tree_serialisation import load_tree


def main(tree_path, number_of_nucleotids):
    alphabet = ['A', 'C', 'T', 'G']
    alphabetSize = len(alphabet)

    nbState = 4
    # transition matrix of the toy gene finder
    A = np.zeros((nbState, nbState))
    A[0, 1] = 1
    A[1, 2] = 1
    A[2, 3] = 0.011
    A[2, 0] = 1 - A[2, 3]
    A[3, 3] = 0.33  # 0.9999  # unrealistic ...
    A[3, 0] = 1 - A[3, 3]

    # state initial probability
    b = np.array([0.25, 0.25, 0.26, 0.24])
    # load the phylogenetic model from JSON
    tree = load_tree(tree_path)
    trees = []

    for j in range(nbState):
        trees.append(scale_branches_length(tree))

    animalNames = ["dog", "cat", "pig", "cow", "rat", "mouse", "baboon",
                   "human"]
    """[...], such as the higher average rate of substitution and the greater
    transition/transversion ratio, in noncoding and third-codon-position sites
    than in firstand second- codon-position sites[...]"""

    pi = np.zeros((nbState, alphabetSize))
    # substitution rates for pi 0 and 1 are between 0 and 0.001
    pi[0] = np.random.rand(alphabetSize) * 0.001
    pi[1] = np.random.rand(alphabetSize) * 0.001
    # but between 0 and 0.01 for pi 2 and 3
    pi[2] = np.random.rand(alphabetSize) * 0.01
    pi[3] = np.random.rand(alphabetSize) * 0.01
    pi /= pi.sum(axis=1)[:, None]

    # translation/transversion rate
    kappa = np.array([2.3, 2.7, 4.3, 5.4])

    strands, states = generate_case(A, b, pi, kappa,
                                    trees, number_of_nucleotids)
    print(strands)


def generate_case(A, b, pi, kappa,
                  trees, number_of_nucleotids):
    """
    Generate a test case with DNA strands and the list of Ground truth states
    Args :
           - A (np array : nbState, nbState) state transition matrix
           - b (np vector: alphabetSize) initial discrete probability
               distribution for the states
            - pi (nparray : nbState, alphabetSize) nucleotids background
            frequencies, state dependent
           - kappa (np vector: size nbState) translation transversion rate
           - trees (list of dicts) list of phylogenetic trees
           - number_of_nucleotids (int) number of nucleotid in each strand
    returns :
           - strands (list of np vector uint8: number_of_nucleotids)
               list of the sequence of nucleotids for each taxa
           - states (np vector uint8: number_of_nucleotids) list of ground
               truth states for each sie
    """
    Q = rate_sub_HKY(pi, kappa)
    # generation

    # GT states
    states = generate_gt_state(A, b, number_of_nucleotids)
    # initial values
    X = generate_initial_vector(pi, states)

    strands = evolution(X, states, trees, Q)

    return strands, states


def scale_branches_length(tree, scale=1):
    '''Given a tree in the dictionnary of list format it returns another
    tree with randomised branches length
        Args:
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - scale (float, optionnal) : scaling factor applied to branch
            lengths
        Output:
            - tree with same shape but scaled branches length
    '''
    tree_cp = deepcopy(tree)

    def rescale_node(node):
        children = tree_cp[node]
        if children:
            for child in children:
                new_child = child["node"]
                rescale_node(new_child)
                child["branch"] *= scale

    rescale_node(max(tree_cp.keys()))
    return tree_cp


def rate_sub_HKY(pi, kappa):
    """ define the rate substitution matrices according to the HKY model for
    all states
    Args :
            - pi (nparray : nbState, alphabetSize) nucleotids background
            frequencies, state dependent
           - kappa (np vector, size nb states) translation/transversion rate
    returns : Q (np array, nb states x alphabetSize x alphabetSize)
    the rate substituon matrices for a states
    """
    nbState = len(kappa)
    alphabetSize = pi.shape[1]
    Q = np.zeros((nbState, alphabetSize, alphabetSize))
    for j in range(nbState):
        for i in range(alphabetSize):
            Q[j, i, :] = pi[j]
            Q[j, i, (i + 2) % alphabetSize] *= kappa[j]
            # put in each diagonal a term such that the rows sums to 0
            Q[j, i, i] -= np.sum(Q[j, i, :])
    return Q


def generate_initial_vector(pi, states):
    '''Return a random vector of nucleotids as integers
        Args:
            - pi (nparray : nbState, alphabetSize) nucleotids background
            frequencies, state dependent
            - states (nparray: nbNucleotids), vector of the state for each site
        Output:
            - np.vector with values between 0 and size(b)-1
            follows distribution b
    '''
    nbState, alphabetSize = pi.shape
    nbNucleotids = states.shape[0]

    cumsum = np.cumsum(pi, axis=1)
    random_values = np.random.rand(nbNucleotids)
    X = np.zeros(nbNucleotids, dtype=np.uint8)
    # now let us draw according to a discrete law in a vectorial way
    for i in range(alphabetSize):
        X[random_values < cumsum[states, i]] = i
        # we erase values that are lower than cumsum[i] to prevent the
        # corresponding nucleotid to be overwritten at the following step
        random_values[random_values < cumsum[states, i]] = 1

    return X


def generate_gt_state(A, b, nbNucleotids):
    '''Use the state transition matrix A to generate of state path
        Args:
            - A (np matrix) state transition matrix
            - b (array of float, sums to 1) initial discrete distribution of
            states
            - nbNucleotids (int) length of the DNA in Nucleotids
        Output:
            - np. vector of int from 0 to nbState-1
    '''
    states = np.empty(nbNucleotids, dtype=np.uint8)

    nbState = A.shape[0]
    # the first one is drawn with the law b
    discrete_law = np.cumsum(b)

    x = np.random.rand(1)[0]

    index = 0
    while x > discrete_law[index]:
        index += 1
    states[0] = index
    for i in range(nbNucleotids-1):
        # draw the next state using the state transition matrix
        discrete_law = np.cumsum(A[states[i]])

        x = np.random.rand(1)[0]

        index = 0
        while x > discrete_law[index]:
            index += 1
        states[i+1] = index
    return states


def evolution(X, states, trees, Q):
    '''Use a vector of DNA X and make it evolve by running it through a
    phylogenetic tree Q
        Args:
            - X (np vector):  nucleotids as integers
            - trees (dict): list of trees as dictionnaries
            - states (np vector): state path
            - Q (narray ): list of substitution rate matrix
        Output:
            - tree with same shape but randomised branches length
    '''
    nbState = Q.shape[0]
    alphabetSize = Q.shape[1]

    def evolve(node, strand):
        children = trees[0][node]
        if children:
            res = []
            for c in range(len(children)):
                new_Q = np.zeros_like(Q)
                # compute probability matrices for every state for l&r branches
                for j in range(nbState):
                    new_br = trees[j][node][c]["branch"]
                    new_Q[j] = expm(new_br * Q[j])

                new_strand = np.zeros_like(strand)
                # the new strand is drown randomly from the previous one
                # using the probability matrix
                cumsum = np.cumsum(new_Q, axis=2)
                random_values = np.random.rand(strand.shape[0])
                # vectorial discrete draw
                for i in range(alphabetSize):
                    new_strand[random_values < cumsum[states, strand, i]] = i
                    random_values[
                        random_values < cumsum[states, strand, i]] = 1

                new_child = children[c]["node"]
                res += evolve(new_child, new_strand)
            return res
        else:
            return [strand]
    return evolve(max(trees[0].keys()), X)


def parse_args():
    parser = argparse.ArgumentParser("Data generation script")

    parser.add_argument("tree_path",
                        help="path of a JSON file encoding a tree")
    parser.add_argument("-n", "--number_of_nucleotids", type=int,
                        help="number of genrated nucleotids for each taxa")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args.tree_path, args.number_of_nucleotids)
