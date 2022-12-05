import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def hypothesis(theta0, theta1, X):
    return theta0 + (theta1 * X)

def cost_function(theta0, theta1, X, y):
    m = X.shape[0]
    return (1 / (2 * m)) * np.sum(np.power((hypothesis(theta0, theta1, X) - y) , 2))

def derivative_theta0(theta0, theta1, X, y):
    m = X.shape[0]
    val = (1 / m) * np.sum(hypothesis(theta0, theta1, X) - y)
    return val

def derivative_theta1(theta0, theta1, X, y):
    m = X.shape[0]
    val = (1 / m) * np.sum((hypothesis(theta0, theta1, X) - y) * X)
    return val

def train_model(theta0, theta1, learning_rate, epoch, X_train, y_train, X_test, y_test, progress_bar):
    history = { 'train_loss' : [], 'test_loss'  : [] }
    train_loss = None
    test_loss  = None
    
    for i in range(epoch):
        temp0 = derivative_theta0(theta0, theta1, X_train, y_train)
        temp1 = derivative_theta1(theta0, theta1, X_train, y_train)
        
        theta0 = theta0 - (learning_rate * temp0)
        theta1 = theta1 - (learning_rate * temp1)
        
        train_loss = cost_function(theta0, theta1, X_train, y_train)
        test_loss  = cost_function(theta0, theta1, X_test, y_test)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        val1 = i + 1
        val2 = epoch // 100
        if val1 % val2:
            progress_bar.progress((val1 // val2) + 1)
    
    history['epoch'] = epoch
    history['learning_rate'] = learning_rate
    history['final_training_loss'] = train_loss
    history['final_testing_loss']  = test_loss
    history['theta0'] = theta0
    history['theta1'] = theta1
        
    return history