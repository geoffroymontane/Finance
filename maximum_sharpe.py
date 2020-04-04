import random
import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

global_Cov = np.array([])
global_E = np.array([])

def cov_rend_moy_N():

	data = pd.DataFrame()
	path = "data_bourse/"

	for filename in os.listdir(path):
		temp = pd.read_csv(path + filename)
		data[filename[:3]] = (temp[filename[:3] + "~high"] + temp[filename[:3] + "~low"])/2

	new_columns = []

	for series in [ data[col] for col in data.columns]:
		new_series = []
		for i in range(len(series) - 3):
			new_series.append((series[i+3] - series[i])/series[i])
		new_columns.append(new_series)

	res = pd.DataFrame(new_columns)

	res = res.transpose()

	return res.cov(),res.mean()


def compute_sharpe_ratio(Cov, E, W):

    n = 0
    d = 0

    for i in range(len(W)):
        n += W[i] * E[i]

    for i in range(len(W)):
        for j in range(len(W)):
            d += W[i] * W[j] * Cov[i][j]

    d = d ** 0.5

    return n / d

def sharpe_ratio_(W):

    global global_E
    global global_Cov

    return -compute_sharpe_ratio(global_Cov, global_E, W)

def coordinate_descent_monte_carlo(Cov, E, iterations):

    N = len(E)
    W = [1 / N for i in range(N)]

    ratio = compute_sharpe_ratio(Cov, E, W)

    for k in range(iterations):
        
        indices = [i for i in range(N)]
        random.shuffle(indices)

        for index in indices:

            W_test = W.copy()

            for i in range(100):
                y = random.random()
                W_test[index] = y

                new_ratio = compute_sharpe_ratio(Cov, E, W_test)
                if new_ratio > ratio:
                    W[index] = y
                    ratio = new_ratio
        
        print(ratio)

    s = 0
    for i in range(len(W)):
        s += W[i] 
    for i in range(len(W)):
        W[i] /= s

    return W, compute_sharpe_ratio(Cov, E, W)


def nelder_mead_simplex(Cov, E):

    global global_Cov
    global global_E
    global_Cov = Cov
    global_E = E 

    N = len(Cov)
    W = [1 / N for i in range(N)]
    W = minimize(sharpe_ratio_, W, method='nelder-mead', 
                   options={'disp': True}).x


    s = 0
    for i in range(len(W)):
        s += W[i] 
    for i in range(len(W)):
        W[i] /= abs(s)

    return W, compute_sharpe_ratio(Cov, E, W) 


def bfgs_method(Cov, E):
    
    global global_Cov
    global global_E
    global_Cov = Cov
    global_E = E 

    N = len(Cov)
    W = [1 / N for i in range(N)]
    W = minimize(sharpe_ratio_, W, method='BFGS', 
                   options={'disp': True}).x


    s = 0
    for i in range(len(W)):
        s += W[i] 
    for i in range(len(W)):
        W[i] /= abs(s)

    return W, compute_sharpe_ratio(Cov, E, W) 


Cov, E = cov_rend_moy_N()

print("SIMPLEX")
print("--------------------")
initial = datetime.now()
#W, sharpe_ratio = nelder_mead_simplex(Cov, E)
W, sharpe_ratio = coordinate_descent_monte_carlo(Cov, E, 20)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(W)
print("Ratio :")
print(sharpe_ratio)

print("")
print("")

print("BFGS")
print("--------------------")
initial = datetime.now()
W, sharpe_ratio = bfgs_method(Cov, E)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(W)
print("Ratio :")
print(sharpe_ratio)

print("")
print("")
