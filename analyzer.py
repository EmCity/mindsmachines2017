import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

#with open('train.csv') as csvfile:
#	reader = csv.reader(csvfile, delimiter=';', quotechar="'")
#
#	for row in reader:
#		print(row[1])

time_columns = ['date_reception_OMP','date_besoin_client',
	'date_transmission_proc','date_emission_commande','date_prev_livraison_prog',
	'date_reelle_livraison_prog','date_reelle_livraison_indus','date_prev_fin_usinage',
	'date_reelle_fin_usinage','date_prev_fin_TS','date_reelle_fin_TS','date_expedition',
	'date_livraison_contractuelle','date_livraison_previsionnelle_S','date_reception_effective',
	'date_livraison_contractuelle_initiale','date_liberation','date_affectation']

import pandas as pd
df = pd.read_csv('data/train.csv',  parse_dates=time_columns)


## Iterate over all columns and drop sparse ones
for column in df:
    percent_nan = float(pd.isnull(df[column]).sum())/len(df[column])
    # print(column + " " + str(float(pd.isnull(df[column]).sum())/len(df[column])))
    if percent_nan > 0.1:
        df = df.drop(column, 1)
        print("dropped column: " + column)


# for leftover nans, replyce with average
for column in df:
    try:
        df[column].fillna((df[column].mean()), inplace=True)
    except:
    	most_common = df[column].value_counts().index[0]
        df[column].fillna(most_common)
        print("replaced nans with: " + str(most_common))

#convert timestamps


for column in time_columns:
	df[column] = pd.to_datetime(df[column], unit='s')
	#for in range(len(df)):

df.to_csv("cleaned_output.csv", sep=";")

#wirte to csv
y = df['date_reception_OMP']
X = df.drop('date_reception_OMP',axis=1)


#model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.8)

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print(str(regr.score(X_test,y_test)))


