import numpy as np


def viterbi(S, A, b, E):
    '''
    Viterbi algorithm
    Input:
        S : state space (line vector 1,...,M)
        A : matrix of state-transition probabilities (M rows, M columns)
        b : vector of initial-state probabilities (dimension M)
        E : matrix of emission probabilities computed with Felstenstein algorithm (M rows, L columns)
        (instead of E => call directly Felstenstein in this function?)
    Output:
        phi : the maximum-likelihood path (vector of dimension L)
    '''

    # Initialization
    M, L = E.shape
    Mu = np.zeros((M, L))
    Mu[:, 0] = b*E[:, 0]
    argmaxMu = np.zeros((M, L), dtype=int)
    phi = np.zeros(L)
    # Recursion
    for t in range(1, L):
        for s in range(M):
            prob = A[:, s]*Mu[:, t-1]
            Mu[s, t] = E[s, t]*max(prob)
            argmaxMu[s, t] = np.argmax(prob)
    phi[-1] = S[np.argmax(Mu[:, -1])]
    # Backward pass
    for t in range(L-2, -1, -1):
        phi[t] = S[argmaxMu[phi[t+1], t+1]]
    return phi
