#Import Numpy as np and Pandas as pd.
import numpy as np
import pandas as pd

#Reading and storing training and testing data from csv files into Pandas dataframes.
train = pd.read_csv('train.data',header = None)
test = pd.read_csv('test.data', header = None)

#Function definition for training a model with input data X and output data Y, using regularization parameter.
def train_model(X, Y, reg_lambda, num_iters):

  #Initializing weights and Bias with zeros.
    W = np.zeros(X.shape[1], dtype=np.float128)
    B = 0

     #Iterating over the terms for a given number of iterations with each element in the training data.
    for i in range(num_iters):
        for j in range(X.shape[0]):
          
          #Calculating the activation score for the jth sample in X using the weights W and bias B.
          #checking for misclassification create a prediction using the sign function and compare it to the actual label Y[j].
            a = np.dot(X[j], W) + B
            y_pred = np.sign(a)
            if y_pred != Y[j]:
                y_hat = np.sign(np.dot(W, X[i]))
                Res = reg_lambda * W* (1-y_hat)

                #Checking for misclassification, create a prediction using the sign function and compare it to the actual label Y[j].
                W = W + Y[j] * X[j] + Res
                B = B + Y[j]
    return W, B

#Defining a function to test a model using input X, output Y, weights W and bias B.
def test_model(X, Y, W, B):

  #Calculating activation score using dot product of input and weights, and adding bias term.
    a = np.dot(X, W) + B
    y_pred = np.sign(a)

    #Calculates the accuracy given by the model on the test data and returns it.
    accuracy = np.mean(y_pred == Y)
    return accuracy

#Extracting input and output variables from training and testing datasets.
X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values

#Getting the unique value from Y_train through np.unique
classes = np.unique(Y_train)
#To implement matrix dimension by getting the length of unique values from Y_train.
num_classes = len(classes)

#Converting class labels into binary class for each classification model.
Y_12_train = np.where(Y_train == 'class-1', 1, -1)
Y_23_train = np.where(Y_train == 'class-2', 1, -1)
Y_13_train = np.where(Y_train == 'class-3', 1, -1)
Y_12_test = np.where(Y_test == 'class-1', 1, -1)
Y_23_test = np.where(Y_test == 'class-2', 1, -1)
Y_13_test = np.where(Y_test == 'class-3', 1, -1)

#Weights and bias are updated accordingly for the given iterations.
W_12, B_12 = train_model(X_train, Y_12_train, 1, 20)
W_23, B_23 = train_model(X_train, Y_23_train, 1, 20)
W_13, B_13 = train_model(X_train, Y_13_train, 1, 20)

#Calculating its accuracies by testing the model.
train_accr_12 = test_model(X_train, Y_12_train, W_12, B_12)
train_accr_23 = test_model(X_train, Y_23_train, W_23, B_23)
train_accr_13 = test_model(X_train, Y_13_train, W_13, B_13)
test_accr_12 = test_model(X_test, Y_12_test, W_12, B_12)
test_accr_23 = test_model(X_test, Y_23_test, W_23, B_23)
test_accr_13 = test_model(X_test, Y_13_test, W_13, B_13)

#Calculating the accuracies of each classes

print("Class 1 vs Class 2: Train Acc = {:.2f} %".format(train_accr_12*100))
print("Class 2 vs Class 3: Train Acc = {:.2f} %".format(train_accr_23*100))
print("Class 1 vs Class 3: Train Acc = {:.2f} %".format(train_accr_13*100))
print("Class 1 vs Class 2: Test Acc = {:.2f} %".format(test_accr_12*100))
print("Class 2 vs Class 3: Test Acc = {:.2f} %".format(test_accr_23*100))
print("Class 1 vs Class 3: Test Acc = {:.2f} % \n".format(test_accr_13*100))

print("******** 1 vs Rest approach************\n")

#Choosing the best accuracy and lambda value to start with.
#Tune the overall impact of the regularization term by multiplying its value by a scalar known as lambda (also called the regularization rate).
best_accuracy = 0
best_reg_lambda = None

# To avoid runtime error float128 is used
for reg_lambda in [0.01, 0.1, 1.0, 10.0, 100.0]:

  # getting the length of unique values from Y_train to implement matrix dimension
    num_classes=len(np.unique(Y_train))
    W = np.zeros((num_classes, X_train.shape[1]), dtype=np.float128)
    B = np.zeros(num_classes, dtype=np.float128)
    for i, class_label in enumerate(classes):
        Y_train_class = np.where(Y_train == class_label, 1, -1)
        W[i], B[i] = train_model(X_train, Y_train_class, 1, 20)

    # Iterating over each lambda values to determine the accuracies of each classes

    train_accr = np.zeros(num_classes)
    test_accr = np.zeros(num_classes)
    for i, class_label in enumerate(classes):
        Y_train_class = np.where(Y_train == class_label, 1, -1)
        Y_test_class = np.where(Y_test == class_label, 1, -1)
        train_accr[i] = test_model(X_train, Y_train_class, W[i], B[i])
        test_accr[i] = test_model(X_test, Y_test_class, W[i], B[i])
        print("{} vs Rest: Train Acc = {:.2f} with coeff:{:.2f} % ".format(class_label, train_accr[i]*100, reg_lambda))
        print("{} vs Rest: Test Acc = {:.2f} with coeff:{:.2f} % ".format(class_label, test_accr[i]*100, reg_lambda))

    #Getting the average accuracy and also the lambda value with maximum accuracy
    average_accuracy = (test_accr_12 + test_accr_23 + test_accr_13) / 3
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_reg_lambda = reg_lambda
      


print("Best regularization parameter: {:.3f}".format(best_reg_lambda))
print("Best average test accuracy: {:.2f} %\n".format(best_accuracy * 100))


print("************L2 Regularisation applied************\n")

#Choosing the best accuracy and lambda value to start with.
#Tune the overall impact of the regularization term by multiplying its value by a scalar known as lambda (also called the regularization rate).
best_accuracy = 0
best_reg_lambda = None

#Given the values iterate for lambda.
for reg_lambda in [0.01, 0.1, 1.0, 10.0, 100.0]:

    #Weights and bias are updated accordingly for the given iterations.    
    W_12, B_12 = train_model(X_train, Y_12_train, reg_lambda, 20)
    W_23, B_23 = train_model(X_train, Y_23_train, reg_lambda, 20)
    W_13, B_13 = train_model(X_train, Y_13_train, reg_lambda, 20)

    #Calculating its accuracies by testing the model.
    train_accr_12 = test_model(X_train, Y_12_train, W_12, B_12)
    train_accr_23 = test_model(X_train, Y_23_train, W_23, B_23)
    train_accr_13 = test_model(X_train, Y_13_train, W_13, B_13)
    test_accr_12 = test_model(X_test, Y_12_test, W_12, B_12)
    test_accr_23 = test_model(X_test, Y_23_test, W_23, B_23)
    test_accr_13 = test_model(X_test, Y_13_test, W_13, B_13)

    print("Class 1 vs Class 2: Train Acc = {:.2f} % with coeff:{:.2f}".format(train_accr_12*100, reg_lambda))
    print("Class 2 vs Class 3: Train Acc = {:.2f} % with coeff:{:.2f}".format(train_accr_23*100, reg_lambda))
    print("Class 1 vs Class 3: Train Acc = {:.2f} % with coeff:{:.2f}".format(train_accr_13*100, reg_lambda))
    print("Class 1 vs Class 2: Test Acc = {:.2f} % with coeff:{:.2f}".format(test_accr_12*100, reg_lambda))
    print("Class 2 vs Class 3: Test Acc = {:.2f} % with coeff:{:.2f}".format(test_accr_23*100, reg_lambda))
    print("Class 1 vs Class 3: Test Acc = {:.2f} % with coeff:{:.2f}".format(test_accr_13*100, reg_lambda))

    
    #Getting the average accuracy and also the lambda value with maximum accuracy
    average_accuracy = (test_accr_12 + test_accr_23 + test_accr_13) / 3
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_reg_lambda = reg_lambda
      


print("Best regularization parameter: {:.3f}".format(best_reg_lambda))
print("Best average test accuracy: {:.2f} %\n".format(best_accuracy * 100))
