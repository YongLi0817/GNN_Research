import numpy as np
import scipy as sp
from numpy import linalg as LA
import seaborn as sns
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
np.random.seed(42)

def ERO(r, eta, p):
    n = r.shape[0]
    r = r.reshape(n, 1)
    M = np.max(np.abs(r))
    e = np.ones((n, 1))
    H_star = eta * p * (r.dot(e.T) - e.dot(r.T))
    H = r.dot(e.T) - e.dot(r.T)
    # Generate all random indicators at once
    indicators = np.triu(np.random.choice(3, (n, n), p=[eta*p, p*(1-eta), 1-p]),1)
    # Generate random values for the second case
    Z = np.random.uniform(-M, M, (n, n))
    H[indicators == 2] = 0
    H[indicators == 1] = Z[indicators == 1]
    H = np.triu(H, 1) - np.triu(H, 1).T
    delta = H - H_star
    return H, delta

def circular_shift(x, shift):
    return np.concatenate((x[-shift:], x[:-shift]))

# def cyclic_perm(x,shift):
#     # n = len(a)
#     b = [[x[i - j] for i in range(shift)] for j in range(shift)]
#     return b


def outer_product(x, y):
    return np.outer(x, y)

def hadamard_product(x, y):
    return x * y

def upset(s, C, shift):
    # print(ones)
    sigma_s = circular_shift(s, shift)
    n = len(s)
    sigma_s = sigma_s.reshape(n,1)
    sigma_outer_ones_T = outer_product(sigma_s, np.ones(len(sigma_s)))
    ones_outer_sigma_T = outer_product(np.ones(len(sigma_s)), sigma_s.T)
    print(np.ones(len(sigma_s)).shape, sigma_s.T.shape)
    term1 = sigma_outer_ones_T - ones_outer_sigma_T
    # print([C != 0])
    # print(hadamard_product(term1, [C!=0]))
    # result = hadamard_product(term1, A)
    result = hadamard_product(term1, [C!=0])
    # P = (sigma_s @ ones.T - ones @ sigma_s.T) * [C!=0]
    # print(P)    P and result are the same --  above calculation should be correct
    print(np.sign(result) - np.sign(C))
    print(np.sum(np.sign(result) - np.sign(C)))
    return 0.5 * (np.sum( np.abs(np.sign(result) - np.sign(C))))
    # return 0.5 * np.max(np.sum(np.sign(result) - np.sign(C),axis = 0))
    # return np.max(np.sum(np.abs(P - C),axis = 0))

    ## ALternative error 
    # print(result - C)
    # print(np.sum(result - C))
    # return np.sum(result - C)


def syncrank(C,g, type):
    n = C.shape[0]
    Theta = 2 * np.pi * g * C / (n-1)
    # print(Theta)
    H = np.exp(1j * Theta)
    if type == 'spectral':
        d = np.sum(np.abs(H),axis = 1) + 1e-10
        ##  d200‘200’ -- array
        ##  ERO model -- all nodes have edges, thus degree = shape[0]
        v,psi = eigs(1j * np.diag(d**(-1)) @ H,1,which = 'LR')
        # v,psi = eigs(1j * np.diag(H**(-1)) @ H,1,which = 'LR')
        xi = np.abs(v)[0]
        # print(v,xi)
        r_hat =  psi / np.abs(psi)
        r_hat = r_hat.reshape(n)
        psi = psi.reshape(n)
        display(psi)
        angles_radians = np.angle(psi)
        display(angles_radians)
        angle_hat = np.angle(r_hat)
        angles_modulo_2pi = np.mod(angles_radians, 2 * np.pi)
        angles_degrees = np.degrees(angles_modulo_2pi)

        # return angle
        # angle[angle < 0] += 2 * np.pi
        # angle_hat[angle_hat < 0] += 2 * np.pi
        # s = np.argsort(angle) # (n,)
        print('angle_degrees is ',angles_degrees)
        sorted_indices = np.argsort(angles_degrees)
        print('label of degree from smallest to largest',sorted_indices)
        # s = pd.Series(angle_hat.tolist()).rank()
        # s = np.array(s).reshape(n)
        # print(s)
        obj = 9999999999999
        best_shift = 0
        for shift in range(n):
            obj1 = upset(sorted_indices,C,shift)
            print(obj1)
            if obj1 < obj:
                obj = obj1
                best_shift = shift
                # print(circular_shift(s,best_shift))
        return circular_shift(sorted_indices,best_shift)
