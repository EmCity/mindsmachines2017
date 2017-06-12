import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, svm
import pandas as pd
import numpy as np

time_columns = ['date_reception_OMP_new', 'date_besoin_client_new', 'date_transmission_proc_new',
                'date_emission_commande_new', 'date_livraison_contractuelle_new', 'date_livraison_previsionnelle_S_new',
                'date_reception_effective_new', 'date_livraison_contractuelle_initiale_new', 'date_liberation_new',
                'date_affectation_new']

df = pd.read_csv("cleaned_output.csv", sep=";", parse_dates=time_columns)

for column in df:
    if df[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

y = df['total_cycle_duration_new']
X = df.drop('total_cycle_duration_new', axis=1)
X = X.drop('date_liberation_new', axis=1)
X = X.drop('date_reception_OMP_new', axis=1)

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
for i in range(0, len(regr.coef_)):
    print(str(X_train.columns.values.tolist()[i]) + " " + str(regr.coef_[i]))
pred_result = regr.predict(X_test)

result_df  = pd.concat([X_test, y_test], axis=1)
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

# svr = svm.SVR(kernel='linear')
# svr.fit(X_train, y_train)
# print("svr regression " + str(svr.score(X_test, y_test)))
# print("svr regression " + str(math.sqrt(metrics.mean_squared_error(y_test, svr.predict(X_test)))))
