'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import numpy as np
import math
from sklearn import datasets, linear_model, model_selection, preprocessing, metrics, svm
import pandas as pd

# Parameters
learning_rate = 0.01
training_epochs = 20
display_step = 5

# Training Data
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
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
X = X.drop('date_liberation_new', axis=1)
X = X.drop('date_reception_OMP_new', axis=1)

# model_selection.TimeSeriesSplit
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
test_X = X_test.values
test_Y = y_test.values
train_X = X_train.values
train_Y = y_train.values
print('train_X: ',train_X.shape, type(train_Y.shape))
print('test_X: ',test_X.shape, type(test_X.shape))
print('test_Y',test_Y.shape, type(test_Y.shape))
print('train_Y',train_Y.shape, type(train_Y.shape))
print('test_Y',np.reshape(test_Y, (1291,1)).shape, type(test_Y.shape))
print('train_Y',np.reshape(train_Y, (3012,1)).shape, type(train_Y.shape))
#print(np.reshape(test_Y, (1291,1)).shape)
#print(np.reshape(train_Y, (3012,1)).shape)
n_samples = train_X.shape[0]
print(n_samples)

test_Y = np.reshape(test_Y, (1291,1))
train_Y = np.reshape(train_Y, (3012,1))

# tf Graph Input
#X = tf.placeholder("float", [len(X_train.index),54])
#Y = tf.placeholder("float", [len(X_train.index),1])
X = tf.placeholder(tf.float32, shape=(len(X_train.index), 54))
Y = tf.placeholder("float", shape=(len(X_train.index), 1))
# Set model weights
W = tf.Variable(tf.random_normal([54,1]), name="weight")
#b = tf.Variable(rng.randn(), name="bias")
b = tf.Variable(tf.random_normal([1]))
# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)
#print(pred)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

acost = tf.reduce_sum(tf.pow(pred-Y, 2)) * (tf.sigmoid(Y-pred)*100.0+1.0)/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(acost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            #c = sess.run(acost, feed_dict={X: train_X, Y:train_Y})
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    #training_cost = sess.run(acost, feed_dict={X: train_X, Y: train_Y})
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    #plt.plot(train_X, train_Y, 'ro', label='Original data')
    #plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    #plt.legend()
    #plt.show()

    # Testing example, as requested (Issue #2)
    #test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    #test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    
    print("Testing... (Mean square loss Comparison)")
    #testing_cost = sess.run(
    #    tf.reduce_sum(tf.pow(pred - Y, 2)) * (tf.sigmoid(Y-pred)*100.0+1.0) / (2 * test_X.shape[0]),
    #    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    #plt.show()