#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

def train_data(kernel_type, features_train, labels_train, c_value):
    clf = SVC(C = c_value, kernel = kernel_type) 
    t0 = time()
    clf.fit(features_train,labels_train)
    print("training time:", round(time()-t0, 3), "s")
    return clf

 
def predict_data(clf,features_test, labels_test):
    t1 = time()
    pred=clf.predict(features_test)
    print("prediction time:", round(time()-t1, 3), "s")
    acc = accuracy_score(pred, labels_test)
    print("The Accuracy is",acc)
    return pred


def get_filter():
    print('Hello! Let\'s explore SVM behaviour')
    data_choices = ['small', 'full']
    kernel_choices = ['rbf', 'linear']

    
    data_choice = input("Would you like to see SVM behaviour for full size or small size dataset. Please enter Full or Small ").lower()
    while data_choice not in data_choices:
        data_choice = input("Would you like to see SVM behaviour for full size or small size dataset. Please enter Full or Small ").lower()
    
    if data_choice == 'full':
        data_full = True
    elif data_choice == 'small':
        data_full = False
    
    kernel_choice = input("Would you like to see SVM behaviour for linear or rbf kernel \n").lower()
    while kernel_choice not in kernel_choices:
        kernel_choice = input("Would you like to see SVM behaviour for linear or rbf kernel \n").lower()
    
    
    c_value = input("Please Enter the value of C \n").lower()
    if c_value.isdigit():
        c = int(c_value)
            
    while c_value.isdigit() == False:
        c_value = input("Please Enter a correct value of C \n").lower()
        if c_value.isdigit():
            c = int(c_value)
            break
     
    return kernel_choice, data_full, c
   

   
def main():
    while True:
         kernel_type, data_full,c = get_filter()
         features_train, features_test, labels_train, labels_test = preprocess()

         if data_full:
             print('Training and predicting full data size and {} kernel'.format(kernel_type))
             clf = train_data(kernel_type, features_train, labels_train, c)
             predictions = predict_data(clf,features_test, labels_test)
         else:
            print('Training and predicting samll data size and {} kernel'.format(kernel_type))
            features_train = features_train[:int(len(features_train)/100)]
            print(int(len(labels_train)/100))
            print(int(len(labels_train)))      
            labels_train = labels_train[:int(len(labels_train)/100)]
            clf = train_data(kernel_type, features_train, labels_train, c)
            predictions = predict_data(clf, features_test, labels_test)
         
         #To return the 10th element from the predictions.
         print("The 10th elemnet prediction output =", predictions[10])
         #To return the 26th element from the predictions.
         print("The 26th elemnet prediction output =", predictions[26])
         #To return the 50th element from the predictions.
         print("The 50th elemnet prediction output =", predictions[50])
         #To answer the question how many mail for Chris according to what is expected from SVM.
         print("Chris Predictions emails numbers is", (predictions == 1).sum())
         #To answer the question how many mail for sara according to what is expected from SVM.
         print("Sara Predictions emails numbers is", (predictions == 0).sum())



         restart = input('\nWould you like to restart? Enter yes or no.\n')
         if restart.lower() != 'yes':
             break

if __name__ == "__main__":
	main()
