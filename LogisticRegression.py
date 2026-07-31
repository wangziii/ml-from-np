# Basic Logistic Regression class
# Zi Wang, 2026

import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, features_col, label_col, prediction_col = 'prediction', threshold = 0.5, rate = 0.1, max_iter = 100):
        self.features_col = features_col
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.threshold = threshold
        self.rate = rate
        self.max_iter = max_iter
    
    def sigmoid(self, x):
        '''
        Inline logistic (sigmoid) function
        '''
        return 1. / (1. + np.exp(-x))
    
    def lr_loss(self, x, y, w):
        '''
        Inline function to calculate logistic loss (- log-likelihood)
        '''
        m = len(y)
        y_pred = self.sigmoid(np.dot(x, w))
        loss = (y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred))
        tot_loss = -sum(loss) / m
        
        return tot_loss[0]
    
    def lr_grad_loss(self, x, y, w):
        '''
        Inline function to calculate gradient of log loss
        '''
        m = len(y)
        y_pred = self.sigmoid(np.dot(x, w))
        grad = np.dot(x.transpose(), (y_pred - y)) / m
    
        return grad.astype(float)
    
    def lr_gradient_descent(self, x, y, w0, rate = 0.1, max_iter = 100):
        '''
        Inline function to calculate gradient descent of log loss function
        '''
        w = w0
        loss_array = []
        for i in range(max_iter):
            loss = self.lr_loss(x, y, w)
            loss_array.append(loss)
            
            grad = self.lr_grad_loss(x, y, w)
            w -= rate * grad
        return w, loss_array

    def load_df(self, df):
        '''
        Inline function to load pandas dataframe with defined feature and label columns
        Set initial weights (w0) and intercept (b0)
        '''
        features = df[self.features_col].to_numpy()
        label = df[self.label_col].to_numpy()
        
        rows = features.shape[0]
        cols = features.shape[1] 
        
        b0 = np.ones((rows, 1))
        w0 = np.zeros((cols + 1, 1))
        
        X = np.append(b0, features, axis=1)
        y = label.reshape(rows, 1)
        
        return X, y, w0
    
    def fit(self, df):
        '''
        Update weights based on gradient descent of log loss function
        '''
        X, y, w0 = self.load_df(df)

        self.w, self.loss_array = self.lr_gradient_descent(X, y, w0, self.rate, self.max_iter)

        
    def predict(self, df):
        '''
        Predict results with calculated weights
        '''
        X, y, w0 = self.load_df(df)
        
        y_pred = np.where(self.sigmoid(np.dot(X, self.w)) >= self.threshold, 1, 0)

        df_pred = pd.DataFrame(y_pred, columns = [self.prediction_col])
        df_out = pd.merge(df, df_pred, left_index=True, right_index=True)
    
        return df_out
