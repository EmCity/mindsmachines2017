import pandas as pd
from sklearn import datasets, linear_model
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
x_train = train['coeff_avion']
y_train = train['lag_time']
x_test = test['coeff_avion']
y_test = test['lag_time']
print train['date_liberation']

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

print("Mean squared error: %.2f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))