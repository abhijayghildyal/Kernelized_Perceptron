#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:49:50 2018

@author: abhijay
"""

# Comment
import os
if not os.path.exists('Q14_plots'):
    os.mkdir('Q14_plots')

#os.getcwd()
#os.chdir("/home/abhijay/Documents/ML/hw_1/11632196/")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Separate Y label and X from data
def separate_target_variable(data):
    data_x = data.iloc[:,0:-1]
    data_y = data.iloc[:,-1]
    data_y = data_y.apply(lambda y: -1 if y==" <=50K" else 1).tolist()
    return data_x, data_y

# Make train, dev and test set have same number of features
def make_features_correspond(train_, data_):
    missing_features = set(train_.columns) - set(data_.columns)
    missing_features = pd.DataFrame(0, index=np.arange(len(data_)), columns=missing_features)
    data_ = pd.concat( [data_, missing_features], axis = 1)
    return data_ 

# Make dummy variables
def one_hot_encode(data, categorical_features):    
    for categorical_feature in categorical_features:
        dummies = pd.get_dummies(data[categorical_feature],prefix=categorical_feature)
        dummies = dummies.iloc[:,0:-1] # Remove redundant feature by selecting 0:-1
        data = pd.concat( [data, dummies], axis = 1).drop(categorical_feature, axis=1)
    return data

class Kernelized_Perceptron():
    def __init__(self, train_data, dev_data, test_data):
        
        self.train_accuracy = []
        self.test_accuracy = []
        self.dev_accuracy = []
        self.mistakes_list = []
        
        # Converting numerical data into buckets
        train_data['age'] = train_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        dev_data['age'] = dev_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        test_data['age'] = test_data['age'].apply(lambda x: '0-19' if x>0 and x<20 else ( '20-35' if x>=20 and x<35 else( '35-50' if x>=35 and x<50 else '>50' )))
        
        train_data['hours-per-week'] = train_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        dev_data['hours-per-week'] = dev_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        test_data['hours-per-week'] = test_data['hours-per-week'].apply(lambda x: '0-15' if x>0 and x<15 else ( '15-30' if x>=15 and x<30 else( '30-45' if x>=30 and x<45 else ( '45-60' if x>=45 and x<60 else '>60' ) )))
        
        # Separating out features from target variable
        train_x, self.train_y = separate_target_variable(train_data)
        dev_x, self.dev_y = separate_target_variable(dev_data)
        test_x, self.test_y = separate_target_variable(test_data)
        
        # One hot encode all categorical variables
        categorical_features = [train_x.columns[col] for col, col_type in enumerate(train_x.dtypes) if col_type == np.dtype('O') ]
        #numerical_features = [col for col, col_type in enumerate(train_x.dtypes) if col_type != np.dtype('O') ]
        
        train_x = one_hot_encode(train_x, categorical_features)
        dev_x = one_hot_encode(dev_x, categorical_features)
        test_x = one_hot_encode(test_x, categorical_features)
        
        
        # Make features in dev categories consistent with train
        dev_x = make_features_correspond( train_x, dev_x)
        test_x = make_features_correspond( train_x, test_x)
        
        # train_features = set(train_x.columns[(train_x.var(axis=0)>0.05)].tolist())
        # dev_features = set(dev_x.columns[(dev_x.var(axis=0)>0.1)].tolist())
        # test_features = set(test_x.columns[(test_x.var(axis=0)>0.05)].tolist())
        
        # final_list_features = list(train_features.intersection(test_features))
        
        final_list_features = list(train_x.columns)
        
        final_list_features.sort()
        
        train_x = train_x[final_list_features]
        dev_x = dev_x[final_list_features]
        test_x = test_x[final_list_features]
        
        # Now that the features are consistent I can convert my datasets into numpy arrays
        train_x = np.array(train_x.values)
        dev_x = np.array(dev_x.values)
        test_x = np.array(test_x.values)
        
        # Scale values in the dataset between [0,1]        
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_x.astype(float))
        self.train_x = scaling.transform(train_x)
        self.dev_x = scaling.transform(dev_x)
        self.test_x = scaling.transform(test_x)
        
        #scaler = preprocessing.StandardScaler().fit(train_x)
        #self.train_x = scaler.transform(train_x)
        #self.test_x = scaler.transform(test_x)
        #self.dev_x = scaler.transform(dev_x)
        
    # Polynomial kernel
    def poly_kernel( self, x1, x2, degree):
        return (1 + np.dot(x1, x2)) ** degree
    
    def kernelized_perceptron( self, X, x, y, degree, training=True):
        
        # Initialize
        mistakes = 0
        
        # Kernelize
        k = self.poly_kernel(x, X.T, degree)
        
        # For each example
        for i,(x,y) in enumerate(zip( x, y)):
            
            # Calculate f_x
            f_x = np.sum(k[i,:] * self.alpha)
            
            # Check condition
            if y*f_x <= 0 :
                mistakes+=1
                # Update alphas if we are training
                if training == True:
                    self.alpha[i]+=y
        
        return (mistakes)
    
    def kernelized_perceptron_driver( self, iterations, bestDegree): 
        self.alpha = np.zeros( self.train_x.shape[0])
        for itr in range(iterations):
            # Train
            train_mistakes = self.kernelized_perceptron(self.train_x, self.train_x, self.train_y, bestDegree, True)
            self.train_accuracy.append(100.0*(self.train_x.shape[0]-train_mistakes)/self.train_x.shape[0])
            self.mistakes_list.append(train_mistakes)
            
            # Dev
            dev_mistakes = self.kernelized_perceptron(self.train_x, self.dev_x, self.dev_y, bestDegree, False)
            self.dev_accuracy.append((100.0*(self.dev_x.shape[0]-dev_mistakes))/self.dev_x.shape[0])
            
            # Test
            test_mistakes = self.kernelized_perceptron(self.train_x, self.test_x, self.test_y, bestDegree, False)
            self.test_accuracy.append(100.0*((self.test_x.shape[0]-test_mistakes))/self.test_x.shape[0])
    
    def plot_figures( self):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot( range(len(self.mistakes_list)), self.mistakes_list, label='Training Mistakes')
        ax.set_ylim(min(self.mistakes_list)-100,max(self.mistakes_list)+100)
        plt.xticks( range(len(self.train_accuracy)), [str(deg) for deg in range(len(self.train_accuracy))])
        ax.set_title('Mistakes plot', fontsize=18)
        ax.set_ylabel('Mistakes', fontsize=15)
        ax.set_xlabel('Iteration', fontsize=15)
        ax.legend(loc='lower right')
        fig.savefig('Q14_plots/Q14_KernelizedPerceptron_MistakesPlot.png')
        
        # print (self.train_accuracy)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot( range(len(self.train_accuracy)), self.train_accuracy, label='Training Accuracy')
        ax.plot( range(len(self.train_accuracy)), self.dev_accuracy, label='Dev Accuracy')
        ax.plot( range(len(self.train_accuracy)), self.test_accuracy, label='Test Accuracy')
        # ax.set_ylim(max(self.train_accuracy+self.dev_accuracy+self.test_accuracy)+5, min(self.train_accuracy+self.dev_accuracy+self.test_accuracy)-5)
        plt.xticks( range(len(self.train_accuracy)), [str(deg) for deg in range(len(self.train_accuracy))])
        ax.set_title('Accuracy plot', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=15)
        ax.set_xlabel('Iteration', fontsize=15)
        ax.legend(loc='lower right')
        fig.savefig('Q14_plots/Q14_KernelizedPerceptron_AccuracyPlot.png')
        
if __name__ == "__main__":
    
    print ("\n\n =============== Kernelized Perceptron ===============\n")
    
    # Get data
    col_names = ["age","workclass","education","marital_status","occupation","race","gender","hours-per-week","native-country","salary-bracket"]
    
    train_data = pd.read_csv("income-data/income.train.txt", names = col_names)
    dev_data = pd.read_csv("income-data/income.dev.txt", names = col_names)
    test_data = pd.read_csv("income-data/income.test.txt", names = col_names)
    
    bestDegree = 2 # Based on test accuracy from SVM
    
    # Run kernelized perceptron
    kernelizedPerceptron = Kernelized_Perceptron( train_data, dev_data, test_data)
    kernelizedPerceptron.kernelized_perceptron_driver(5, bestDegree)    
    
    # Plot results
    kernelizedPerceptron.plot_figures()
    print ("Ans 14 Done.....")