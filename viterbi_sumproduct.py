import numpy as np


def forward(A, b, E, mode):
    '''
    Forward pass: computes the alpha-messages associated to the Sum-Product algorithm or Viterbi.
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein algorithm (M rows, L columns)
    :param mode: string to precise how to compute the alpha-messages. Must be 'max' for Viterbi,
            'sum' for Sum-Product
    :return: if 'max', matrix of alpha-messages and the argmax matrix ; if 'sum', matrix of alpha-messages
    '''
    if mode != 'max' and mode != 'sum':
        return "Error: Input parameter mode must be 'sum' or 'max'!"
    else:
        # Initializaton
        M, L = E.shape
        alpha = np.zeros((M, L))
        alpha[:, 0] = b*E[:, 0]
        argmaxAlpha = np.zeros((M, L), dtype=int)  # useful for mode 'max' only
        # Recursion
        for t in range(1, L):
            for s in range(M):
                prob = A[:, s]*alpha[:, t - 1]
                if mode == 'sum':
                    prob = sum(prob)
                else:
                    argmaxAlpha[s, t] = np.argmax(prob)
                    prob = max(prob)
                alpha[s, t] = E[s, t]*prob
        return alpha if mode == 'sum' else [alpha, argmaxAlpha]


def backward(A, E):
    '''
    Backward pass: computes the beta-messages for the Sum-Product algorithm
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param E: matrix of emission probabilities computed with Felstenstein algorithm (M rows, L columns)
    :return: matrix of beta-messages
    '''
    # Initialization
    M, L = E.shape
    beta = np.zeros((M, L))
    beta[:, -1] = 1
    # Recursion
    for t in range(L-2, -1, -1):
        for s in range(M):
            beta[s, t] = A[s, :].dot(E[:, t+1]*beta[:, t+1])
    return beta


def sum_product(A, b, E):
    '''
    Sum-Product algorithm (forward-backward procedure)
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein algorithm (M rows, L columns)
    :return: matrix of posterior probabilities
    '''
    M, L = E.shape
    post_probas = np.zeros((M, L))
    # Forward and backward procedure to compute the alpha and beta messages
    alpha = forward(A, b, E, 'sum')
    beta = backward(A, E)
    # Posterior probabilities computation
    for t in range(L):
        post_probas[:, t] = (alpha[:, t]*beta[:, t]) / (alpha[:, t].dot(beta[:, t]))
    return post_probas


def viterbi(S, A, b, E):
    '''
    Viterbi algorithm.
    :param S: state space (line vector 1,...,M)
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein algorithm (M rows, L columns)
                (instead of E => call directly Felstenstein in this function?)
    :return: the maximum-likelihood path (vector of dimension L)
    '''
    # Initialization
    M, L = E.shape
    phi = np.zeros(L)
    # Recursion
    [alpha, argmaxAlpha] = forward(A, b, E, 'max')
    # Decoding the argmax
    opt_sequence = np.zeros(L, dtype=int)
    opt_sequence[-1] = np.argmax(alpha[:, -1])
    phi[-1] = S[opt_sequence[-1]]
    for t in range(L-2, -1, -1):
        opt_sequence[t] = argmaxAlpha[opt_sequence[t+1], t+1]
        phi[t] = S[opt_sequence[t]]
    return phi


## TEST
S=np.array([1,2,3])
A = np.array([[0.2, 0.4, 0.4], [0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
b = np.array([0.2, 0.3, 0.5])
E = np.array([[0.2, 0.6, 0.5], [0.1,  0.1, 0.4], [0.7,  0.3, 0.1]])
path = viterbi(S, A, b, E)
pp = sum_product(A, b, E)