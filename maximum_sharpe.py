import random
import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

global_Cov = np.array([])
global_E = np.array([])
global_test = []
global_names = []
global_test_begin = ""
global_test_end = ""

def cov_rend_moy_N():

    global global_test
    global global_names
    global global_test_begin
    global global_test_end

    data = {}
    data_df = pd.DataFrame()
    path = "data_bourse/"

    sizes = []

    for filename in os.listdir(path):
        temp = pd.read_csv(path + filename)

        for i in range(temp.shape[0]):
            date = temp.loc[i, "date"] 

            if date in data:
                data[date][filename[:3]] = ((temp.loc[i, filename[:3] + "~high"]
                             + temp.loc[i, filename[:3] + "~low"]) / 2)
            else:
                data[date] = {}
                data[date][filename[:3]] = ((temp.loc[i, filename[:3] + "~high"]
                             + temp.loc[i, filename[:3] + "~low"]) / 2)

    global_names = ["EEM", "EFA", "EWJ", "USO", "IWM", "SPY"] 
    for date in data:
        for t in global_names:
            if t not in data[date]:
                data[date] = None
                break
    data_keys = sorted(list(data.keys()))
    for i in reversed(range(len(data_keys))):
        if data[data_keys[i]] == None:
            data_keys.pop(i)

    # data_df est l'échantillon d'apprentissage
    data_df = pd.DataFrame(columns = global_names)
    for i in range(len(data_keys) // 2):

        temp = []
        for e in global_names:
            temp.append(data[data_keys[i]][e])
        data_df.loc[i] = temp

    # global_test est l'échantillon de test 
    global_test_begin = data_keys[len(data_keys) // 2]
    global_test_end = data_keys[len(data_keys) - 1]
    for i in range(len(data_keys) // 2, len(data_keys)):
        temp = []
        for e in global_names:
            temp.append(data[data_keys[i]][e])
        global_test.append(temp)

    new_columns = []
    for series in [data_df[col] for col in data_df.columns]:
        new_series = []
        for i in range(len(series) - 3):
            new_series.append((series[i + 3] - series[i]) / series[i])
        new_columns.append(new_series)

    res = pd.DataFrame(new_columns)
    res = res.transpose()

    return res.cov(), res.mean()


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
        W[i] /= abs(max(W))

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


    max_W = max(W)
    for i in range(len(W)):
        W[i] /= abs(max_W)

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

    max_W = max(W)
    for i in range(len(W)):
        W[i] /= abs(max_W)

    return W, compute_sharpe_ratio(Cov, E, W) 



def test_portofolio(W):

    print("Début " + str(global_test_begin))

    initial = 0
    for i in range(len(W)):
        initial += W[i] * global_test[0][i]
    
    final = 0
    for i in range(len(W)):
        final += W[i] * global_test[len(global_test) - 1][i]

    print(str((final - initial) / initial * 100) + "%")
    print("Fin " + str(global_test_end))


Cov, E = cov_rend_moy_N()

print("SIMPLEX")
print("--------------------")
initial = datetime.now()
W, sharpe_ratio = nelder_mead_simplex(Cov, E)
#W, sharpe_ratio = coordinate_descent_monte_carlo(Cov, E, 20)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(global_names)
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
print(global_names)
print(W)
print("Ratio :")
print(sharpe_ratio)
print("PERFORMANCE")
print("--------------------")
test_portofolio(W)

print("")
print("")
