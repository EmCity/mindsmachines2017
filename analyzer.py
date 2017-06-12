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

#wirte to csv

df.to_csv("cleaned_output.csv", sep=";")

