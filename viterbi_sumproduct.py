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
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param b: vector of initial-state probabilities (dimension n_states)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
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
        n_states, n_sites = E.shape
        alpha_log = np.zeros((n_states, n_sites))
        alpha_log[:, 0] = np.log(b) + np.log(E[:, 0])
        alpha_argmax = np.zeros((n_states, n_sites), dtype=int)  # useful for mode 'max' only
        # Recursion
        for t in range(1, n_sites):
            for s in range(n_states):
                prob = np.log(A[:, s]) + alpha_log[:, t - 1]
                if mode == 'sum':
                    prob = sum_log(prob)
                else:
                    alpha_argmax[s, t] = np.argmax(prob)
                    prob = max(prob)
                alpha_log[s, t] = np.log(E[s, t]) + prob
        return alpha_log if mode == 'sum' else [alpha_log, alpha_argmax]


def backward(A, E):
    '''
    Backward pass: computes the logarithms of beta-messages for the Sum-Product
    algorithm
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
    :return: matrix of log beta-messages
    '''
    # Initialization
    n_states, n_sites = E.shape
    beta_log = np.zeros((n_states, n_sites))
    # Recursion
    for t in range(n_sites-2, -1, -1):
        beta_log[:, t] = sum_log(
            np.log(A) + np.log(E[:, t+1]) + beta_log[:, t+1], axis=1)
    return beta_log


def sum_product(A, b, E):
    '''
    Sum-Product algorithm (forward-backward procedure)
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param b: vector of initial-state probabilities (dimension n_states)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
    :return: matrix of posterior probabilities
    '''
    n_states, n_sites = E.shape
    post_probas = np.zeros((n_states, n_sites))
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
    :param S: state space (line vector 1,...,n_states)
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param b: vector of initial-state probabilities (dimension n_states)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
                (instead of E => call directly Felstenstein in this function?)
    :return: the maximum-likelihood path (vector of dimension n_sites)
    '''
    # Initialization
    n_states, n_sites = E.shape
    phi = np.zeros(n_sites)
    # Recursion
    [alpha_log, alpha_argmax] = forward(A, b, E, 'max')
    # Decoding the argmax
    opt_sequence = np.zeros(n_sites, dtype=int)
    opt_sequence[-1] = np.argmax(alpha_log[:, -1])
    phi[-1] = S[opt_sequence[-1]]
    for t in range(n_sites-2, -1, -1):
        opt_sequence[t] = alpha_argmax[opt_sequence[t+1], t+1]
        phi[t] = S[opt_sequence[t]]
    return phi
