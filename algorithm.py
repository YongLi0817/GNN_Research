import numpy as np
import scipy as sp
from numpy import linalg as LA
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
def upset(s, C, shift):
    ones = np.ones(len(s))
    sigma_s = circular_shift(s, shift)
    print(sigma_s)
    P = (np.outer(sigma_s, ones) - np.outer(ones, s))
    return 0.5 * np.sum(np.abs(np.sign(P) - np.sign(C)))

def syncrank(C,g, type):
    n = C.shape[0]
    Theta = 2 * np.pi * g * C /(n-1)
    H = np.exp(1j * Theta)
    if type == 'spectral':
        d = np.sum(np.abs(H),axis = 1) + 1e-10
        v,psi = eigs(1j * np.diag(d**(-1)) @ H,1,which = 'LR')
        xi = np.abs(v)[0]
        psi = psi.reshape(n)
        angle = np.angle(v)
        angle[angle < 0] += 2 * np.pi
        s = np.argsort(angle) # (n,)
        obj = 99999
        best_shift = 0
        for shift in range(n):
            obj1 = upset(s,C,shift)
            if obj1 < obj:
                obj = obj1
                best_shift = shift
        return circular_shift(s,best_shift)








def spectral_ranking(H):
    n = H.shape[0]
    v,phi = eigs(1j * H,1,which = 'LR')
    sigma = np.abs(v)[0]

    ones = np.ones(n)
    phi = phi.reshape(n)
    # print(phi.shape)
    if np.dot(np.imag(phi),ones) <= 0:
        theta = np.arctan(np.dot(np.real(phi),ones) / np.dot(np.imag(phi),ones))
    else:
        theta = np.arctan(np.dot(np.real(phi),ones) / np.dot(np.imag(phi),ones)) + np.pi
    u2_hat = np.real(phi * np.exp(1j * theta)) / LA.norm(np.real(phi * np.exp(1j * theta)))
    u1_hat = np.imag(phi * np.exp(1j * theta)) / LA.norm(np.real(phi * np.exp(1j * theta)))
    return sigma, u2_hat, u1_hat

def normalized_spectral_ranking(H):
    n = H.shape[0]
    d = np.sum(np.abs(H),axis = 1) + 1e-10
    v,psi = eigs(1j * np.diag(d**(-1)) @ H,1,which = 'LR')
    xi = np.abs(v)[0]

    dones = np.diag(d) @ np.ones(n)
    psi = psi.reshape(n)
    if np.dot(np.imag(psi),dones) <= 0:
        theta = np.arctan(np.dot(np.real(psi),dones) / np.dot(np.imag(psi),dones))
    else:
        theta = np.arctan(np.dot(np.real(psi),dones) / np.dot(np.imag(psi),dones)) + np.pi

    u2_hat = np.real(np.diag(d) @ psi * np.exp(1j * theta)) / LA.norm(np.real(np.diag(d) @ psi * np.exp(1j * theta)))
    u1_hat = np.real(np.diag(d) @ psi * np.exp(1j * theta)) / LA.norm(np.real(np.diag(d) @ psi * np.exp(1j * theta)))
    return xi, u2_hat,u1_hat, d

def kendall(s1,s2):
    n = len(s1)
    tau = 0
    for i in range(n):
        for j in range(i+1,n):
            tau += np.sign(s1[i]-s1[j]) * np.sign(s2[i]-s2[j])
    return 2*tau/(n*(n-1))

def displacement_error(u,v, error_type):
    n = len(u)
    displacement_errors = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if u[j] > u[i] and v[j] < v[i]:
                displacement_errors[i] += 1
            elif u[j] < u[i] and v[i] < v[j]:
                displacement_errors[i] += 1
    if error_type == 'max':
        return np.max(displacement_errors) / (n-1)
    elif error_type == 'total':
        return np.sum(displacement_errors) / (n*(n-1))
        
