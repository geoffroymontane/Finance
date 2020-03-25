import random
import pandas as pd
import os
import numpy as np


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


def sharpe_ratio(Cov, E, W):

    n = 0
    d = 0

    for i in range(len(W)):
        n += W[i] * E[i]

    for i in range(len(W)):
        for j in range(len(W)):
            d += W[i] * W[j] * Cov[i][j]

    d = d ** 0.5

    return n / d



def coordinate_descent_monte_carlo(Cov, E, iterations):

    N = len(E)
    W = [1 / N for i in range(N)]

    ratio = sharpe_ratio(Cov, E, W)

    for k in range(iterations):
        
        indices = [i for i in range(N)]
        random.shuffle(indices)

        for index in indices:

            W_test = W.copy()

            for i in range(100):
                y = random.random()
                W_test[index] = y

                new_ratio = sharpe_ratio(Cov, E, W_test)
                if new_ratio > ratio:
                    W[index] = y
                    ratio = new_ratio
        
        print(ratio)

    return W, sharpe_ratio(Cov, E, W)


Cov, E = cov_rend_moy_N()
W, sharpe_ratio = coordinate_descent_monte_carlo(Cov, E, 100)

print(W, sharpe_ratio)
