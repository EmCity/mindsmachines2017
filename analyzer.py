import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
#with open('train.csv') as csvfile:
#	reader = csv.reader(csvfile, delimiter=';', quotechar="'")
#
#	for row in reader:
#		print(row[1])

import pandas as pd
df = pd.read_csv('train.csv')
lags = df.lag_time
print(lags)