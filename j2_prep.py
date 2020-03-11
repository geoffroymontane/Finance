import pandas as pd
import os


def cov_N():

	N = 0

	data = pd.DataFrame()
	path = "data_bourse/"

	for filename in os.listdir(path):
		temp = pd.read_csv(path + filename)
		data[filename[:3]] = (temp[filename[:3] + "~high"] + temp[filename[:3] + "~low"])/2

	return data.cov(),N



data,N = cov_N()

print(data)

