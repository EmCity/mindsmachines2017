import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, svm
import numpy as np
import pandas as pd


def normalize_time(input_time):
    delta_time = pd.Series(pd.datetime(1970, 1, 1))
    return (input_time - delta_time).dt.total_seconds()


time_columns = ['date_reception_OMP', 'date_besoin_client',
                'date_transmission_proc', 'date_emission_commande', 'date_prev_livraison_prog',
                'date_reelle_livraison_prog', 'date_reelle_livraison_indus', 'date_prev_fin_usinage',
                'date_reelle_fin_usinage', 'date_prev_fin_TS', 'date_reelle_fin_TS', 'date_expedition',
                'date_livraison_contractuelle', 'date_livraison_previsionnelle_S', 'date_reception_effective',
                'date_livraison_contractuelle_initiale', 'date_liberation', 'date_affectation']

df = pd.read_csv('data/test.csv', parse_dates=time_columns)

# remove same collums as in train.csv



to_drop_frame = ['longueur_piece_brut_mm', 'largeur_piece_brut_mm', 'epaisseur_brut_mm', 'debit_mm2',
                 'codes_protections', 'complexite_piece_pp', 'indice_client', 'factory', 'usine', 'priorite',
                 'advancement_percentage_tooling', 'date_prev_livraison_outillage', 'date_reelle_livraison_outillage',
                 'advancement_percentage_program', 'date_prev_livraison_prog', 'date_reelle_livraison_prog',
                 'advancement_percentage_industrialization', 'date_prev_livraison_indus', 'date_reelle_livraison_indus',
                 'type_lot', 'date_prev_fin_usinage', 'date_reelle_fin_usinage', 'lieu_TS', 'date_prev_fin_TS',
                 'date_reelle_fin_TS', 'date_prev_reception_QI_fournisseur', 'date_reelle_reception_QI_fournisseur',
                 'lieu_equipement', 'date_prev_fin_equipement', 'date_reelle_fin_equipement', 'date_expedition',
                 'quantite_expediee', 'ecart_commande', 'date_creation_nomenclature_appro',
                 'date_reception_matiere_fournisseur', 'date_liberation_previsionelle', 'quantite_liberee']

## Iterate over all columns and drop sparse ones
for column in to_drop_frame:
    df = df.drop(column, 1)
    if column in time_columns:
        time_columns.remove(str(column))


# for leftover nans, replyce with average
def fillnulls(input_data, replace_value):
    if pd.isnull(input_data):
        return replace_value
    else:
        return replace_value


for column in df:
    try:
        df[column].fillna((df[column].mean()), inplace=True)
        df[column].apply(lambda x: fillnulls(x, df[column].mean()))
    except:
        most_common = df[column].value_counts().index[0]
        df[column].fillna(most_common)
        df[column].apply(lambda x: fillnulls(x, most_common))
        print("replaced nans with: " + str(most_common))

# calculate total_cycle_duration
df['total_cycle_duration'] = df['date_liberation'] - df['date_reception_OMP']
df['delay_liberation'] =  df['date_liberation'] - df['date_livraison_previsionnelle_S']

# convert timestamps
delta_time = pd.Series(pd.datetime(1970, 1, 1))
time_columns.append('total_cycle_duration')
time_columns.append('delay_liberation')

for column in time_columns:
    print(str(column))
    if column == 'total_cycle_duration':
        for i in range(len(df)):
            df.set_value(i, column + "_new", (df.loc[i,column] / np.timedelta64(1, 's')) / (60*60*24))
        df = df.drop(column, 1)
    elif column == 'delay_liberation':
        for i in range(len(df)):
            df.set_value(i, column + "_new", (df.loc[i,column] / np.timedelta64(1, 's')) / (60*60*24))
        df = df.drop(column, 1)
    else:
        for i in range(len(df)):
            df.set_value(i, column + "_new", str(pd.Timestamp(df.loc[i,column]).value))
        df = df.drop(column, 1)


# remove row which empty date_liberation or date_reception_OMP
# df = df[df.date_reception_OMP.notnull()]

# write to csv
df.to_csv("cleaned_output_test.csv", sep=";")
