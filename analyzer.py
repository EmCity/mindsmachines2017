import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
#with open('train.csv') as csvfile:
#	reader = csv.reader(csvfile, delimiter=';', quotechar="'")
#
#	for row in reader:
#		print(row[1])

import pandas as pd
df = pd.read_csv('data/train.csv')


# df.dropna
lags = df.lag_time

NRC_standard_total = df.NRC_standard_total

for i in range(len(df)):
	if pd.isnull(NRC_standard_total[i]) or  pd.isnull(lags[i]) :
		NRC_standard_total.pop(i)
		lags.pop(i)
		print("dropped" + str(i))

#NRC_standard_total.dropna(how='any')


# Split the data into training/testing sets
lags_train = lags[:-1000]
lags_test = lags[-1000:]

# Split the data into training/testing sets
NRC_standard_total_train = NRC_standard_total[:-1000]
NRC_standard_total_test = NRC_standard_total[-1000:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(NRC_standard_total_train, lags_train)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(NRC_standard_total_test) - lags_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(NRC_standard_total_test, lags_test))

# Plot outputs
plt.scatter(NRC_standard_total_test, lags_test,  color='black')
plt.plot(NRC_standard_total_test, regr.predict(NRC_standard_total_test), color='blue',
         linewidth=3)