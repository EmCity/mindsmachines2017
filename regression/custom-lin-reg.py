
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, svm
import pandas as pd

class GradientDescent():
    def __init__(self, alpha=0.1, tolerance=0.02, max_iterations=500):
        #alpha is the learning rate or size of step to take in 
        #the gradient decent
        self._alpha = alpha
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        #thetas is the array coeffcients for each term
        #the y-intercept is the last element
        self._thetas = None

    def alpha(self, y_pred, ys):
        #define custom alpha
        if(ys.all() - y_pred.all() > 0):
            return 100.0
        else:
            return 1.0

    def fit(self, xs, ys):
        num_examples, num_features = np.shape(xs)
        self._thetas = np.ones(num_features)

        xs_transposed = xs.transpose()
        for i in range(self._max_iterations):
            print "weights" , self._thetas
            hypot = np.dot(xs,self._thetas)
            diffs = hypot - ys
            #sum of the squares
            #difference between hypothesis and actual values
            cost = np.sum(diffs**2) / (2*num_examples)
            #print ys - hypot
            #cost = np.sum(np.square(ys - hypot) * self.alpha(hypot, ys))
            #calculate averge gradient for every example
            gradient = np.dot(xs_transposed, diffs) / num_examples
            #gradient = ( np.dot(xs_transposed, (hypot - ys)) * self.alpha(hypot, ys) ) / num_examples
            #gradient = np.dot(xs_transposed, (hypot - ys) * self.alpha(hypot, ys) * 2) / num_examples
            #gradient = (self.alpha(hypot, ys) *  np.dot(xs_transposed, ys) - np.dot(xs_transposed, (hypot - ys)) ) / np.square(hypot)
            #update the coeffcients
            self._thetas = self._thetas-self._alpha*gradient
            
            #check if fit is "good enough"
            if cost < self._tolerance:
                return self._thetas
        diffs_new = np.dot(xs,self._thetas) - ys    
        print "Mean squared error", np.sum(diffs_new**2) / (2*num_examples)
        return self._thetas

    def predict(self, x):
        prediction = np.dot(x, self._thetas)
        #num_examples, num_features = np.shape(xs)
        #print "This is the mean_square_error: ",  / num_examples
        return prediction

#load some example data
#data = np.loadtxt("iris.data.txt", usecols=(0,1,2,3), delimiter=',')
"""
data = pd.read_csv('')
col_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

data_map = dict(zip(col_names, data.transpose()))

#create martix of features
features = np.column_stack((data_map['petal length'], np.ones(len(data_map['petal length']))))

gd = GradientDescent(tolerance=0.022)
thetas = gd.fit(features, data_map['petal width'])
gradient, intercept = thetas

#predict values accroding to our model 
ys = gd.predict(features)

plt.scatter(data_map['petal length'], data_map['petal width'])
plt.plot(data_map['petal length'], ys)
plt.show()"""

time_columns = ['date_reception_OMP_new', 'date_besoin_client_new', 'date_transmission_proc_new',
                'date_emission_commande_new', 'date_livraison_contractuelle_new', 'date_livraison_previsionnelle_S_new',
                'date_reception_effective_new', 'date_livraison_contractuelle_initiale_new', 'date_liberation_new',
                'date_affectation_new']

df = pd.read_csv("../cleaned_output.csv", sep=";", parse_dates=time_columns)

for column in df:

    if df[column].dtype == object:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

y = df['total_cycle_duration_new']
X = df.drop('total_cycle_duration_new', axis=1)

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

#create matrix of features
features = X_train.values
print features
print y_train.values
print y_train
print type(y_train.values)


gd = GradientDescent(tolerance=0.022)
thetas = gd.fit(X_train.values[:20], y_train.values[:20])
#gradient, intercept = thetas

#predict values accroding to our custom model 
ys = gd.predict(features)
"""

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
pred_result = regr.predict(X_test)

result_df = pd.concat([X_test, y_test], axis=1)
pred_df = pd.DataFrame(data=pred_result,index=result_df.index, columns=['total_cycle_duration_predict'])
result_df = pd.concat([result_df, pred_df], axis=1)
#result_df.to_csv(".results/result_lin_reg.csv", sep=',')

print("linear regression " + str(regr.score(X_test, y_test)))
print("linear regression " + str(math.sqrt(metrics.mean_squared_error(y_test, regr.predict(X_test)))))

svr = svm.SVR()
svr.fit(X_train, y_train)
print("svr regression " + str(svr.score(X_test, y_test)))
print("svr regression " + str(math.sqrt(metrics.mean_squared_error(y_test, svr.predict(X_test)))))

"""
