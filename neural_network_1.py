"""
 CSV file result classification with legitimate = 1, Phishing = -1 loaded
 Using sklearn library of python to train neural network
"""
import numpy
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
datatrain = pd.read_csv('dataset.csv')
datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

# Spliting and specify testing data = 0.4
X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:30],
                                                    datatrain_array[:,30],
                                                    test_size=0.4)

"""
input layer has 30 neuron because 30 feature in phishing dataset 1
hidden layer has 10 neuron
output layer has 2 neuron because 2 class in phishing dataset
using inbuilt solver sgd = stochastic gradient descent
learning rate = 0.01
iteration = 1500
"""

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),activation='logistic', solver='sgd',learning_rate_init=0.01,max_iter=1500,random_state=7,tol=0.001)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print("testing model score")
print(mlp.score(X_test,y_test))

