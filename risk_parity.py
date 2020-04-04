import random
import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

global_Cov = np.array([])

def cov_N():

	N = 0

	data = pd.DataFrame()
	path = "data_bourse/"

	for filename in os.listdir(path):
		temp = pd.read_csv(path + filename)
		data[filename[:3]] = ((temp[filename[:3] + "~high"]
                             + temp[filename[:3] + "~low"]) / 2)

	return np.array(data.cov())


def f(Cov, W):

    S = 0

    N = len(Cov)

    a = 0
    for i in range(N):
        for j in range(N):
            a += Cov[i][j] * W[i] * W[j]

    for k in range(N):
        
        b = 0
        for i in range(N):
            b += W[i] * Cov[k][i]

        b *= N
        S += (W[k] - a / b) ** 2

    return S / (a / b) ** 2

def f_(W):
    return f(global_Cov, W)

def compute_risk_contributions(Cov, W):
    risk_contributions = []

    for i in range(len(W)):
        
        n = 0
        for k in range(len(W)):
            n += Cov[i][k] * W[k]
        n *= W[i]

        risk_contributions.append(n)

    return risk_contributions


def nelder_mead_simplex(Cov, precision):

    global global_Cov

    N = len(Cov)
    W = [1 / N for i in range(N)]
    global_Cov = Cov
    W = minimize(f_, W, method='nelder-mead', 
                   options={'xatol': precision, 'disp': True}).x
    ref = []
    for k in range(N):
        a = 0
        for i in range(N):
            for j in range(N):
                a += Cov[i][j] * W[i] * W[j]

        b = 0
        for i in range(N):
            b += W[i] * Cov[k][i]

        b *= N

        ref.append(a / b)

    return W, compute_risk_contributions(Cov, W), ref, f(Cov, W) 


def bfgs_method(Cov):
    
    global global_Cov
    global_Cov = Cov

    N = len(Cov)
    W = [1 / N for i in range(N)]
    W = minimize(f_, W, method='BFGS', 
                   options={'disp': True}).x
    ref = []
    for k in range(N):
        a = 0
        for i in range(N):
            for j in range(N):
                a += Cov[i][j] * W[i] * W[j]

        b = 0
        for i in range(N):
            b += W[i] * Cov[k][i]

        b *= N

        ref.append(a / b)

    return W, compute_risk_contributions(Cov, W), ref, f(Cov, W) 

def coordinate_descent_monte_carlo(Cov, precision):

    N = len(Cov)
    W = [1 / N for i in range(N)]

    error = f(Cov, W)

    while error > precision:
        
        indices = [i for i in range(N)]
        random.shuffle(indices)

        for index in indices:

            for i in range(200):
                y_test = random.random()
                y = W[index]
                W[index] = y_test

                new_error = f(Cov, W)
                if new_error < error:
                    error = new_error
                else:
                    W[index] = y
        print(error)

    s = 0
    for i in range(len(W)):
        s += W[i] 
    for i in range(len(W)):
        W[i] /= s


    ref = []
    for k in range(N):
        a = 0
        for i in range(N):
            for j in range(N):
                a += Cov[i][j] * W[i] * W[j]

        b = 0
        for i in range(N):
            b += W[i] * Cov[k][i]

        b *= N

        ref.append(a / b)

    return W, compute_risk_contributions(Cov, W), ref, f(Cov, W)


def check_risk_parity(Cov, W):

    sigma = []
    N = len(Cov)

    for i in range(N):

        s = 0
        for k in range(N):
            s += Cov[k][i] * W[k]

        s *= W[i]
        sigma.append(s)

        m = 0
        for v in sigma:
            m += v
        m = m / len(sigma)

        e = 0
        for v in sigma:
            e += v ** 2
        e = e / len(sigma) - m ** 2

    return "Error : " + str(e / m * 100) + "%"


Cov = cov_N()

print("SIMPLEX")
print("--------------------")
initial = datetime.now()
W, risk_contributions, ref, error = nelder_mead_simplex(Cov, 1e-5)
#W, risk_contributions, ref, error = coordinate_descent_monte_carlo(Cov, 0.01)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(W)
print("Objectif :")
print(ref)
print("Risk contributions :")
print(risk_contributions)
print("Erreur :")
print(error)

print("")
print("")

print("BFGS")
print("--------------------")
initial = datetime.now()
W, risk_contributions, ref, error = bfgs_method(Cov)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(W)
print("Objectif :")
print(ref)
print("Risk contributions :")
print(risk_contributions)
print("Erreur :")
print(error)

print("")
print("")

