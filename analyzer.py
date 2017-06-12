import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection, preprocessing


# with open('train.csv') as csvfile:
#	reader = csv.reader(csvfile, delimiter=';', quotechar="'")
#
#	for row in reader:
#		print(row[1])

def normalize_time(input_time):
    delta_time = pd.Series(pd.datetime(1970, 1, 1))
    return (input_time - delta_time).dt.total_seconds()


time_columns = ['date_reception_OMP', 'date_besoin_client',
                'date_transmission_proc', 'date_emission_commande', 'date_prev_livraison_prog',
                'date_reelle_livraison_prog', 'date_reelle_livraison_indus', 'date_prev_fin_usinage',
                'date_reelle_fin_usinage', 'date_prev_fin_TS', 'date_reelle_fin_TS', 'date_expedition',
                'date_livraison_contractuelle', 'date_livraison_previsionnelle_S', 'date_reception_effective',
                'date_livraison_contractuelle_initiale', 'date_liberation', 'date_affectation']

import pandas as pd

df = pd.read_csv('data/train.csv', parse_dates=time_columns)

## Iterate over all columns and drop sparse ones
for column in df:
    percent_nan = float(pd.isnull(df[column]).sum()) / len(df[column])
    # print(column + " " + str(float(pd.isnull(df[column]).sum())/len(df[column])))
    if percent_nan > 0.1:
        df = df.drop(column, 1)
        if column in time_columns:
            time_columns.remove(str(column))
        print("dropped column: " + column)

# for leftover nans, replyce with average
def fillnulls(input_data, replace_value):
    if pd.isnull(input_data):
        return replace_value
    else:
        return replace_value


for column in df:
    try:
        df[column].fillna((df[column].mean()), inplace=True)
        df[column].apply(lambda x:  fillnulls(x, df[column].mean()))
    except:
        most_common = df[column].value_counts().index[0]
        df[column].fillna(most_common)
        df[column].apply(lambda x:  fillnulls(x, most_common))
        print("replaced nans with: " + str(most_common))

# convert timestamps

delta_time = pd.Series(pd.datetime(1970, 1, 1))

# for column in time_columns:
# 	df[column].apply(normalize_time)
# for row in df[column]:
#	row = (row - delta_time).dt.total_seconds()
# df[column] = pd.to_datetime(df[column], unit='s')
# for in range(len(df)):


# write to csv
df.to_csv("cleaned_output.csv", sep=";")

# label encoding
#
# text_collumn = ['type_reference', 'designation', 'programme', 'code_filiere', 'magasin', 'famille_code_filiere', 'norm_mat_inge	norm_mat_fab', ' indice_manufacturing',
#                 'avion_impacte_pp', 'complexite_piece_pp', 'priorite_pp	indice_client_pp', 'cms_mat_pp', 'fournisseur_cmde',
#                  'supplier_cmde	rang_appro', 'indice_client', 'factory', 'usine', 'type_cycle',

for column in df:

    if df[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

df.to_csv("cleaned_output_labeled.csv", sep=";")

y = df['date_reception_OMP']
X = df.drop('date_reception_OMP', axis=1)

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.8)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(str(regr.score(X_test, y_test)))
