#################################
#            IML_T4             #         
#################################
#
# File Name: mainT4.py
#
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Abdelrahman-Shalaby (shalabya@student.ethz)

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import matplotlib.pyplot as plt


#################################
# 1) Import and preprocess data #       
#################################
train_data_labeled = pd.read_hdf("train_labeled.h5", "train")
train_data_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
y_train_unlabeled = np.genfromtxt('y_train_unlabeled_NN.csv', delimiter=',')
test_data = pd.read_hdf("test.h5", "test")


# Load data
#labeled
y_train_labeled = np.array(train_data_labeled)[:,0]
X_train_labeled = np.array(train_data_labeled)[:,1:]
#unlabeled
y_train_unlabeled = np.array(y_train_unlabeled)
X_train_unlabeled = np.array(train_data_unlabeled)
#test
X_test = np.array(test_data)

# Combine both labeled and unlabeled training data
X_train_labeled = np.concatenate([X_train_labeled,X_train_unlabeled], axis=0)
y_train_labeled = np.concatenate([y_train_labeled,y_train_unlabeled], axis=0)




N = np.shape(X_train_labeled)[1] # number of features in input vector
print("the number of features is {}" .format(N))
U = np.shape(X_train_labeled)[0]
print ("the number of combined data sets is {}" .format(U)) #this should be 30,000
F = np.shape(X_test)[0] # the number of test data sets
print("the number of test data sets is {}" .format(F)) #this should be 8000


# Convert classes to one hot encodings
print(y_train_labeled) # before
lb = preprocessing.LabelBinarizer()
lb.fit(y_train_labeled)
y_train_labeled = lb.transform(y_train_labeled)
L = np.shape(y_train_labeled)[1] # the number of labels
print("the number of labels is {}" .format(L))
print(y_train_labeled) # after

#plt.hist(X_train_labeled[:,0])
#plt.show()

# Mean remove the training and test data
X_train_labeled = (X_train_labeled - np.mean(X_train_labeled, axis=0))/(np.std(X_train_labeled, axis=0))
X_train_unlabeled = (X_train_unlabeled - np.mean(X_train_unlabeled, axis=0))/(np.std(X_train_unlabeled, axis=0))
X_test = (X_test - np.mean(X_test, axis=0))/(np.std(X_test, axis=0))


# Split training data further into training and validation sets
X_train_labeled, X_val_labeled, y_train_labeled, y_val_labeled = train_test_split(X_train_labeled, y_train_labeled, test_size=0.02, random_state=42)

# X_train plot of histogram
#plt.hist(X_train_labeled[:,0])
#plt.show()




#################################
# 2) Set up Network Structure   #       
#################################
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

alpha = 1  # learning rate
inputs = tf.placeholder(tf.float32, (None, N))
y_true = tf.placeholder(tf.float32, (None, L))



def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1) #shape: the shape of the output tensor 
    bias = tf.random_normal((1,shape[1]), stddev=0.1) 
    return tf.Variable(weights), tf.Variable(bias)

# First hidden layer
nhu1 = 400
w1, b1 = init_weights((N, nhu1))
o1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)

nhu2 = 400
w2, b2 = init_weights((nhu1, nhu2))
o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)

nhu3 = 400
w3, b3 = init_weights((nhu2, nhu3))
o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)

#nhu4 = 200
#w4, b4 = init_weights((nhu3, nhu4))
#o4 = tf.nn.relu(tf.matmul(o3, w4) + b4)

#nhu5 = 200
#w5, b5 = init_weights((nhu4, nhu5))
#o5 = tf.nn.relu(tf.matmul(o4, w5) + b5)
#
#nhu6 = 200
#w6, b6 = init_weights((nhu5, nhu6))
#o6 = tf.nn.relu(tf.matmul(o5, w6) + b6)


nhuO = L
wO, bO = init_weights((nhu3, nhuO))
out = tf.nn.relu(tf.matmul(o3, wO) + bO)

# Softmax function
y_hat = tf.nn.softmax(out)
predictions = tf.argmax(y_hat, 1)

# the cost is calculated as the difference between the real and the predicted y vector.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_hat)) 
opt = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
acc = tf.metrics.accuracy(labels=tf.argmax(y_true,1), predictions=tf.argmax(y_hat,1))


#################################
# 3) Train Network!             #       
#################################
sess = tf.Session()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_g)
sess.run(init_l)

k = 50 # number of batches
kf = KFold(n_splits=k, shuffle=False)

epochs = 200
acc_val_vect = []
acc_train_vect = []

for e in range(epochs):
    print("Epoch {}/{}".format(e+1, epochs))
    # First evaluate performance on validation set
    acc_val = sess.run(acc, feed_dict={inputs: X_val_labeled, y_true: y_val_labeled})[0]
    acc_train = sess.run(acc, feed_dict={inputs: X_train_labeled, y_true: y_train_labeled})[0]

    acc_val_vect.append(acc_val)
    acc_train_vect.append(acc_train)

    # Split data into batches and train, SGD
    for _, t_idx in kf.split(X_train_labeled):
        X_b = X_train_labeled[t_idx,:]
        y_b = y_train_labeled[t_idx,:]

        _ = sess.run(opt, feed_dict={inputs:X_b, y_true:y_b})

#Final performance on both sets
acc_val = sess.run(acc, feed_dict={inputs: X_val_labeled, y_true: y_val_labeled})[0]
acc_train = sess.run(acc, feed_dict={inputs: X_train_labeled, y_true: y_train_labeled})[0]


acc_val_vect.append(acc_val)
acc_train_vect.append(acc_train)

e_vect = range(epochs+1)
max_acc_train = np.amax(acc_train)
max_acc_val = np.amax(acc_val)
print (max_acc_train)
print (max_acc_val)
plt.plot(e_vect, acc_train_vect, 'b')
plt.plot(e_vect, acc_val_vect, 'm')
plt.show()



#################################
# 4) Predict labels of test data!#  
#################################

probabilities = sess.run(y_hat, feed_dict={inputs : X_test})
labels = sess.run([predictions], feed_dict={inputs : X_test})
labels = np.reshape(labels,(-1,1))

np.savetxt('probabilities.csv', probabilities, fmt='%f', newline='\n', comments='', delimiter=',')
np.savetxt('labels.csv', labels, fmt='%f', newline='\n', comments='', delimiter=',')





#results_prob = sess.run([y_hat], feed_dict={inputs : X_test})
#results_labels = tf.argmax(results_prob, 1)
#results_labels = np.reshape(results_labels,(-1,1)) 
#print (results_labels)

#T = np.shape(results_prob)
#TT = np.shape(results_labels)
#print (T)
#print (TT)

#ID = np.genfromtxt('ID.csv', delimiter=',')
#ID = np.matrix(ID)
#ID = np.reshape(ID,(-1,1))
#output = np.concatenate((ID, labels), axis=1)

