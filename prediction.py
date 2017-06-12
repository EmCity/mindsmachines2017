import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, svm
import pandas as pd

time_columns = ['date_reception_OMP_new', 'date_besoin_client_new', 'date_transmission_proc_new',
                'date_emission_commande_new', 'date_livraison_contractuelle_new', 'date_livraison_previsionnelle_S_new',
                'date_reception_effective_new', 'date_livraison_contractuelle_initiale_new', 'date_liberation_new',
                'date_affectation_new', 'total_cycle_duration_new']

df = pd.read_csv("cleaned_output.csv", sep=";", parse_dates=time_columns)

for column in df:

    if df[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

y = df['total_cycle_duration_new']
X = df.drop('total_cycle_duration_new', axis=1)

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
pred_result = regr.predict(X_test)

output_prediction = X_test
output_prediction["total_cycle_duration"] = y_test.value
output_prediction["predict total_cycle_duration"] = pred_result.value

output_prediction.to_csv("prediction_result.csv", sep=';')

print("linear regression " + str(regr.score(X_test, y_test)))
print("linear regression " + str(math.sqrt(metrics.mean_squared_error(y_test, regr.predict(X_test)))))

svr = svm.SVR()
svr.fit(X_train, y_train)
print("svr regression " + str(svr.score(X_test, y_test)))
print("svr regression " + str(math.sqrt(metrics.mean_squared_error(y_test, svr.predict(X_test)))))
