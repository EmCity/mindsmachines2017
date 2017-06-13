import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, decomposition
import pandas as pd
import numpy as np


time_columns = ['date_reception_OMP_new', 'date_besoin_client_new', 'date_transmission_proc_new',
                'date_emission_commande_new', 'date_livraison_contractuelle_new', 'date_livraison_previsionnelle_S_new',
                'date_reception_effective_new', 'date_livraison_contractuelle_initiale_new', 'date_liberation_new',
                'date_affectation_new']

df = pd.read_csv("cleaned_output.csv", sep=";", parse_dates=time_columns)

df_test = pd.read_csv("cleaned_output_test.csv", sep=";", parse_dates=time_columns)


for column in df:
    if df[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])



for column in df_test:
    if df_test[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df_test[column])
        df_test[column] = le.transform(df_test[column])



y = df['total_cycle_duration_new']
df = df.drop('total_cycle_duration_new', axis=1)
X = df.drop('date_liberation_new', axis=1)
X = X.drop('date_reception_OMP_new', axis=1)


pca = decomposition.PCA(n_components=3)
pca.fit(X)


for i in range(0, len(pca.explained_variance_ratio_)):
    print(str(X.columns.values.tolist()[i]) + " " + str(pca.explained_variance_ratio_[i]))

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)




pred_result = regr.predict(X_test)

result_df = pd.concat([X_test, y_test], axis=1)
pred_df = pd.DataFrame(data=pred_result, index=result_df.index, columns=['total_cycle_duration_predict'])
result_df = pd.concat([result_df, pred_df], axis=1)
result_df.to_csv("results/result_lin_reg.csv", sep=',')

pred_df = pd.DataFrame(data=pred_result, index=result_df.index, columns=['total_cycle_duration_predict'])
result_df = pd.concat([y_test, pred_df], axis=1)
result_df.to_csv("results/result_only.csv", sep=',')



print("linear regression " + str(regr.score(X_test, y_test)))
print("linear regression " + str(math.sqrt(metrics.mean_squared_error(y_test, regr.predict(X_test)))))

lasso = linear_model.Lasso()
lasso.fit(X_train, y_train)
print("lasso regression " + str(lasso.score(X_test, y_test)))
print("lasso regression " + str(math.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_test)))))

elastic = linear_model.ElasticNet()
elastic.fit(X_train, y_train)
print("elastic regression " + str(elastic.score(X_test, y_test)))
print("elastic regression " + str(math.sqrt(metrics.mean_squared_error(y_test, elastic.predict(X_test)))))


## generation of the test csv result
y_test_data_real = df_test['total_cycle_duration_new']
df_test = df_test.drop('total_cycle_duration_new', axis=1)


regr_trained = linear_model.LinearRegression()
regr_trained.fit(X,y)

X_test_data = df_test.drop('date_liberation_new', axis=1)
X_test_data = X_test_data.drop('date_reception_OMP_new', axis=1)

y_test_data = regr_trained.predict(X_test_data)

test_resuls = []

for i in range(0,len(y_test_data)):
    test_resuls.append([X_test_data.iloc[i]['id_reference'], y_test_data[i]])

print(test_resuls)
np.savetxt('test_deliverable.txt', test_resuls, '%5.4f',delimiter=',')

# print("linear regression " + str(regr_trained.score(X_test_data, y_test_data_real)))
# print("linear regression " + str(math.sqrt(metrics.mean_squared_error(y_test_data_real, regr_trained.predict(X_test_data)))))


