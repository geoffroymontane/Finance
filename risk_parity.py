import random
import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

global_Cov = np.array([])
global_test = []
global_names = []
global_test_begin = ""
global_test_end = ""

def cov_N():

    global global_test
    global global_names
    global global_test_begin
    global global_test_end

    data = {}
    data_iv = {}
    data_df = pd.DataFrame()

    path = "data_bourse/"
    path_iv = "data_bourse_iv/"

    sizes = []

    # Sous-jacent
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

    # IV
    for filename in os.listdir(path_iv):
        temp = pd.read_csv(path_iv + filename)

        for i in range(temp.shape[0]):
            date = temp.loc[i, "date"] 

            if date in data_iv:
                data_iv[date][filename[:3]] = temp.loc[i, "level_W2"]
            else:
                data_iv[date] = {}
                data_iv[date][filename[:3]] = temp.loc[i, "level_W2"]

        
    # Nettoyage des données
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


    data_df = pd.DataFrame(columns = global_names)

    # data_df est l'échantillon d'apprentissage
    for i in range(len(data_keys) // 2):

        temp = []
        for e in global_names:
            temp.append(data[data_keys[i]][e])
        data_df.loc[i] = temp


    # calculer les iv moyens sur l'échantillon d'apprentissage
    iv = [0 for i in range(len(global_names))]
    iv_deno = [0 for i in range(len(global_names))]
    data_iv_keys = sorted(list(data_iv.keys()))
    for i in range(len(data_iv_keys)):
        if data_iv_keys[i] > data_keys[len(data_keys) // 2 - 1]:
            break
        for n in range(len(global_names)):
            if global_names[n] in data_iv[data_iv_keys[i]]:
                iv[n] += data_iv[data_iv_keys[i]][global_names[n]]
                iv_deno[n] += 1

    cov = np.array(data_df.cov())
    for i in range(len(iv)):
        if iv_deno[i] == 0:
            iv[i] = cov[i][i] ** 0.5
        else:
            iv[i] /= iv_deno[i]

    # global_test est l'échantillon de test
    global_test_begin = data_keys[len(data_keys) // 2]
    global_test_end = data_keys[len(data_keys) - 1]
    for i in range(len(data_keys) // 2, len(data_keys)):
        temp = []
        for e in global_names:
            temp.append(data[data_keys[i]][e])
        global_test.append(temp)



    # IV
    cov_iv = np.copy(cov)
    for i in range(len(cov)):
        for j in range(len(cov)):
            cov_iv[i][j] = iv[i] * iv[j] * cov[i][j] / (cov[i][i] * cov[j][j]) ** 0.5

    return cov, cov_iv

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

    max_W = max(W)
    for i in range(len(W)):
        W[i] /= abs(max_W)

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

    max_W = max(W)
    for i in range(len(W)):
        W[i] /= abs(max_W)

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


    max_W = max(W)
    for i in range(len(W)):
        W[i] /= abs(max_W)


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


Cov, Cov_iv = cov_N()

print("SIMPLEX")
print("--------------------")
initial = datetime.now()
W, risk_contributions, ref, error = nelder_mead_simplex(Cov, 1e-5)
#W, risk_contributions, ref, error = coordinate_descent_monte_carlo(Cov, 0.01)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(global_names)
print(W)
print("Objectif :")
print(ref)
print("Risk contributions :")
print(risk_contributions)
print("Erreur :")
print(error)

print("")
print("")

print("BFGS Realised cov")
print("--------------------")
initial = datetime.now()
W, risk_contributions, ref, error = bfgs_method(Cov)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(global_names)
print(W)
print("Objectif :")
print(ref)
print("Risk contributions :")
print(risk_contributions)
print("Erreur :")
print(error)
print("PERFORMANCE")
print("--------------------")
test_portofolio(W)

print("")
print("")

print("BFGS IV")
print("--------------------")
initial = datetime.now()
W, risk_contributions, ref, error = bfgs_method(Cov_iv)
duration = datetime.now() - initial
print("Time:")
print(duration)
print("Poids obtenus :")
print(global_names)
print(W)
print("Objectif :")
print(ref)
print("Risk contributions :")
print(risk_contributions)
print("Erreur :")
print(error)
print("PERFORMANCE")
print("--------------------")
test_portofolio(W)
