import random
import pandas as pd
import os
import numpy as np


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

    for k in range(N):
        
        a = 0
        for i in range(N):
            for j in range(N):
                a += Cov[i][j] * W[i] * W[j]

        b = 0
        for i in range(N):
            b += W[i] * Cov[k][i]

        b *= N
        S += (W[k] - a / b) ** 2

    return S



def coordinate_descent_monte_carlo(Cov, precision):

    N = len(Cov)
    W = [1 / N for i in range(N)]

    error = f(Cov, W)

    while error > precision:
        
        indices = [i for i in range(N)]
        random.shuffle(indices)

        for index in indices:

            W_test = W.copy()

            for i in range(100):
                y = random.random()
                W_test[index] = y

                new_error = f(Cov, W_test)
                if new_error < error:
                    W[index] = y
                    error = new_error
        print(error)

    return W, f(Cov, W)


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
W, error = coordinate_descent_monte_carlo(Cov, 0.00001)
print(check_risk_parity(Cov, W))
