from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score

import numpy as np
"""
 CSV file result classification with legitimate = 1, suspicious = 0, Phishing = -1 loaded
 Using sklearn library of python 
"""

def load_data():    
    training_data = np.genfromtxt('hi.csv', delimiter=',', dtype=np.int32)
    inputs = training_data[:,:-1]
    outputs = training_data[:, -1]
    training_inputs = inputs[:1000]
    training_outputs = outputs[:1000]
    testing_inputs = inputs[1000:]
    testing_outputs = outputs[1000:]
    return training_inputs, training_outputs, testing_inputs, testing_outputs


if __name__ == '__main__':
    print ("Tutorial: Training a logistic regression to detect phishing websites")
    # Decrementing One result column from traning and testing data
    # That column will be used for prediction and finding accuracy
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()
    print ("Training data loaded.")
    classifier = lr()
    print ("Logistic regression classifier created.")

    print ("Beginning model training.")
    classifier.fit(train_inputs, train_outputs)
    print ("Model training completed.")

    
    predictions = classifier.predict(test_inputs)
    print ("Predictions on testing data computed.")
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print ("The accuracy of your logistic regression on testing data is: ")
    print(accuracy)
