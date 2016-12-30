import numpy as np


def sum_log(a, axis=0):
    '''
    Sum when working with logarithms to avoid numerical errors
    :param a: vector or matrix
    :param axis: axis along which the sum is performed, useful if 'a' is a
    matrix
    :return: m + log(sum_i exp(a_i - m)) with m = max(a)
    '''
    # 'a' is a vector
    if a.ndim == 1:
        m = max(a)
        return m + np.log(sum(np.exp(a - m)))
    # 'a' is a matrix
    else:
        m = np.max(a, axis=axis)
        diff = a - m[:, np.newaxis] if axis == 1 else a - m
        return m + np.log(np.sum(np.exp(diff), axis=axis))


def forward(A, b, E, mode):
    '''
    Forward pass: computes the logarithms of alpha-messages associated to the
    Sum-Product
    algorithm or Viterbi. Logarithms are used to avoid numerical errors.
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (M rows, L columns)
    :param mode: string to precise how to compute the log alpha-messages. Must
    be 'max' for Viterbi,
            'sum' for Sum-Product
    :return: if 'max', matrix of log alpha-messages and the argmax matrix ; if
    'sum', matrix of log alpha-messages
    '''
    if mode != 'max' and mode != 'sum':
        return "Error: Input parameter mode must be 'sum' or 'max'!"
    else:
        # Initialization
        M, L = E.shape
        alpha_log = np.zeros((M, L))
        alpha_log[:, 0] = np.log(b) + np.log(E[:, 0])
        argmaxAlpha = np.zeros((M, L), dtype=int)  # useful for mode 'max' only
        # Recursion
        for t in range(1, L):
            for s in range(M):
                prob = np.log(A[:, s]) + alpha_log[:, t - 1]
                if mode == 'sum':
                    prob = sum_log(prob)
                else:
                    argmaxAlpha[s, t] = np.argmax(prob)
                    prob = max(prob)
                alpha_log[s, t] = np.log(E[s, t]) + prob
        return alpha_log if mode == 'sum' else [alpha_log, argmaxAlpha]


def backward(A, E):
    '''
    Backward pass: computes the logarithms of beta-messages for the Sum-Product
    algorithm
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (M rows, L columns)
    :return: matrix of log beta-messages
    '''
    # Initialization
    M, L = E.shape
    beta_log = np.zeros((M, L))
    # Recursion
    for t in range(L-2, -1, -1):
        beta_log[:, t] = sum_log(
            np.log(A) + np.log(E[:, t+1]) + beta_log[:, t+1], axis=1)
    return beta_log


def sum_product(A, b, E):
    '''
    Sum-Product algorithm (forward-backward procedure)
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (M rows, L columns)
    :return: matrix of posterior probabilities
    '''
    M, L = E.shape
    post_probas = np.zeros((M, L))
    # Forward and backward procedure to compute the logarithms of alpha and
    # beta messages
    alpha_log = forward(A, b, E, 'sum')
    beta_log = backward(A, E)
    # Posterior probabilities computation using the log method
    post_probas = np.exp(
        alpha_log + beta_log - sum_log(alpha_log + beta_log, axis=0))
    return post_probas


def viterbi(S, A, b, E):
    '''
    Viterbi algorithm.
    :param S: state space (line vector 1,...,M)
    :param A: matrix of state-transition probabilities (M rows, M columns)
    :param b: vector of initial-state probabilities (dimension M)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (M rows, L columns)
                (instead of E => call directly Felstenstein in this function?)
    :return: the maximum-likelihood path (vector of dimension L)
    '''
    # Initialization
    M, L = E.shape
    phi = np.zeros(L)
    # Recursion
    [alpha_log, argmaxAlpha] = forward(A, b, E, 'max')
    # Decoding the argmax
    opt_sequence = np.zeros(L, dtype=int)
    opt_sequence[-1] = np.argmax(alpha_log[:, -1])
    phi[-1] = S[opt_sequence[-1]]
    for t in range(L-2, -1, -1):
        opt_sequence[t] = argmaxAlpha[opt_sequence[t+1], t+1]
        phi[t] = S[opt_sequence[t]]
    return phi
