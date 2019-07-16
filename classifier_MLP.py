#################################
#            IML_T2             #         
#################################
#
# File Name: main.py
#
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz.ch)

import pandas as pd
import numpy as np
import sys
import os
import csv
import itertools
import matplotlib.pyplot as plt
import scipy

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import KFold 

#sys.path.insert(0, "../")
#import pykernels

np.set_printoptions(suppress=True)




# import data
train_data_labeled = pd.read_hdf("train_labeled.h5", "train")
train_data_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test_data = pd.read_hdf("test.h5", "test")

# Load data
y_train_labeled = np.array(train_data_labeled)[:,0]
X_train_labeled = np.array(train_data_labeled)[:,1:]
X_train_unlabeled = np.array(train_data_unlabeled)
X_test = np.array(test_data)
#N = np.shape(X_train_labeled)[1] # number of features in input vector
#print("the number of features is {}" .format(N))
#F = np.shape(X_test)[0] # the number of test data sets
#print("the number of test data sets is {}" .format(F))
#U = np.shape(X_train_unlabeled)[0]
#print ("the number of unlabeled data sets is {}" .format(U))

# get features and data points
d = np.shape(X_train_labeled)[1] # number of parameters (should be 28)
n = np.shape(X_train_labeled)[0] # number of data points 

# Import Test Data
#test = np.genfromtxt('test.csv', delimiter=',')
#test = np.delete(test, 0, 0) # remove first row
#test = np.matrix(test)
#idx = test[:,0]
#X_test = test[:,1:]

# Check for class imbalance
#print("Num 0s: ", np.sum(y==0))
#print("Num 1s: ", np.sum(y==1))
#print("Num 2s: ", np.sum(y==2))
#print("\n")
#total_counts = np.matrix([[np.sum(y==0)],[np.sum(y==1)],[np.sum(y==2)]])

# SVM's are NOT scale invariant, so we need to normalize the input.
# We can either force the numbers to be between 0 and 1, -1 and 1, 
# or do a mean removal / unit variance.
X_train_labeled = (X_train_labeled - np.mean(X_train_labeled, axis=0))/(np.std(X_train_labeled, axis=0))
X_train_unlabeled = (X_train_unlabeled - np.mean(X_train_unlabeled, axis=0))/(np.std(X_train_unlabeled, axis=0))
X_test = (X_test - np.mean(X_test, axis=0))/(np.std(X_test, axis=0))

# Setup an SVM classifier
#
# For this part of the problem, we can play around with the kernels (linear, polynomial, rbf,
# sigmoid, etc...). See: http://scikit-learn.org/stable/modules/svm.html
# Section 1.4.6 for kernel types. I think we really just need to try a bunch of different methods
# and see what works / does the best job in the end. 
k = 10
kf = KFold(n_splits=k, shuffle=False)

# RBF Code w/ Cross Validation

g_vect = np.logspace(-1,1,3)
score_vect = []
for g in g_vect: # iterate over RBF shape parameters
    print("Testing Gamma = ", g)
    score_g = []
    
    for train_idx, test_idx in kf.split(X_train_labeled): # iterate over folds    
        #clf = svm.SVC(C=1, decision_function_shape='ovo', kernel=pykernels.regular.ANOVA(g))
        clf = svm.SVC(C=1, decision_function_shape='ovo', kernel='rbf', gamma=g)
        #clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=g,
                            #batch_size='auto', learning_rate='adaptive', max_iter=500, warm_start=True)

        X_train, X_val = X_train_labeled[train_idx,:], X_train_labeled[test_idx]
        y_train, y_val = y_train_labeled[train_idx], y_train_labeled[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        C = metrics.confusion_matrix(y_val, y_pred)
        score = np.trace(C)/len(y_val)
        print(C)
        #print(score)
        score_g.append(score)
    score_vect.append(np.mean(score_g))

print(score_vect)

plt.plot(g_vect, score_vect, 'b')
plt.xlabel('RBF Regularization Parameter')
plt.ylabel('Mean Accuracy Rate')
plt.xscale('log')
plt.grid()
plt.show()


'''
clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=10,
        batch_size='auto', learning_rate='adaptive', max_iter=500, warm_start=True)
clf.fit(X, y)
y_pred = clf.predict(X)
C = metrics.confusion_matrix(y, y_pred)
score = np.trace(C)/len(y)
print(C)
print(score)
'''

#sys.exit()

# Train on Test Data
y_train_unlabeled = clf.predict(X_train_unlabeled)
y_train_unlabeled = np.reshape(y_train_unlabeled,(-1,1))
output = np.matrix(y_train_unlabeled)
np.savetxt('y_train_unlabeled.csv', output, fmt='%d', newline='\n', comments='', delimiter=',')




