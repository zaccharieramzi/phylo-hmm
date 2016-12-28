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
    A[3, 3] = 0.9999  # unrealistic ...
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

    # translation/transversion rate
    kappa = np.array([2.3, 2.7, 4.3, 5.4])

    Q = rate_sub_HKY(pi, kappa)
    # generation

    # initial values
    X = generate_initial_vector(b, number_of_nucleotids)
    states = generate_gt_state(A, number_of_nucleotids)
    print(evolution(X, states, trees, Q))


def scale_branches_length(tree, scale=0.1):
    '''Given a tree in the dictionnary of list format it returns another
    tree with randomised branches length
        Args:
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - max_amp (float, optionnal) : maximal amplitude of branch length
        Output:
            - tree with same shape but randomised branches length
    '''
    tree_cp = deepcopy(tree)

    def rescale_node(node):
        childs = tree_cp[node]
        if childs:
            for child in childs:
                new_child = child["node"]
                rescale_node(new_child)
                child["branch"] *= scale

    rescale_node(max(tree_cp.keys()))
    return tree_cp


def rate_sub_HKY(pi, kappa):
    """ define the rate substitution matrices according to the HKY model for
    all states
    Args :
           - pi (np vector, size alphabetSize) transition probabilities
           - kappa (np vector, size nb states) translation transversion rate
    returns : Q (np array, nb states x alphabetSize x alphabetSize)
    the rate substituon matrices for a states
    """
    nbState = len(kappa)
    alphabetSize = len(pi)
    Q = np.zeros((nbState, alphabetSize, alphabetSize))
    for j in range(nbState):
        for i in range(alphabetSize):
            Q[j, i, :] = pi[i]
            Q[j, i, (i + 2) % alphabetSize] *= kappa[j]
            # put in each diagonal a term such that the rows sums to 0
            Q[j, i, i] -= np.sum(Q[j, i, :])
    return Q


def generate_initial_vector(b, nbNucleotids):
    '''Return a random vector of nucleotids as integers
        Args:
            - b (array of float, sums to 1) initial discrete distribution
            - nbNucleotids (int), size of the output
        Output:
            - np.vector with values between 0 and size(b)-1
            follows distribution b
    '''
    cumsum = np.cumsum(b)
    random_values = np.random.rand(nbNucleotids)
    X = np.empty(nbNucleotids, dtype=np.uint8)
    for i in range(b.shape[0]):
        X[random_values < cumsum[i]] = i
        random_values[random_values < cumsum[i]] = 1
    return X


def generate_gt_state(A, nbNucleotids):
    '''Use the state transition matrix A to generate of state path
        Args:
            - nbState (np matrix) state transition matrix
            - nbNucleotids (int) length of the DNA in Nucleotids
        Output:
            - np. vector of int from 0 to nbState-1
    '''
    states = np.empty(nbNucleotids, dtype=np.uint8)

    nbState = A.shape[0]
    states[0] = np.random.randint(0, nbState, 1)[0]
    for i in range(nbNucleotids-1):

        discrete_law = A[states[i]]
        discrete_law = np.cumsum(discrete_law)
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

    def evolve(node, vector):
        childs = trees[0][node]
        if childs:
            res = []
            for c in range(len(childs)):
                new_Q = np.zeros_like(Q)
                # compute probability matrices for every state for l&r branches
                for j in range(len(Q)):
                    new_br = trees[j][node][c]["branch"]
                    new_Q[j] = expm(new_br * Q[j])

                new_vector = np.zeros_like(vector)
                for i in range(vector.shape[0]):
                    discrete_law = np.cumsum(new_Q[states[i]][vector[i]])
                    x = np.random.rand(1)[0]
                    index = 0
                    while x > discrete_law[index]:
                        index += 1
                    new_vector[i] = index

                new_child = childs[c]["node"]
                res += evolve(new_child, new_vector)
            return res
        else:
            return [vector]
    return evolve(max(trees[0].keys()), X)


def parse_args():
    parser = argparse.ArgumentParser("Data generation script")

    parser.add_argument("tree_path",
                        help="path of a JSON file encoding a tree")
    parser.add_argument("-n", "--number_of_nucleotids", type=int,
                        help="number of genrated nucleotids for each taxa")
    return parser.parse_args()

if __name__ == "__main__":
    # define the number of images to generate and run the script
    args = parse_args()


    main(args.tree_path, args.number_of_nucleotids)
