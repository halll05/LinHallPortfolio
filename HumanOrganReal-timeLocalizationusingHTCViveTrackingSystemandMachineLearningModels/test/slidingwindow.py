#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score
import random

from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt


from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
import time


import copy as cp

############import data

filenames=['data/nonrigid_rollall_processed.csv', 'data/nonrigid_pitch2_processed.csv',
'data/nonrigid_x4_processed.csv', 'data/nonrigid_y2_processed.csv',  'data/nonrigid_yaw2_processed.csv',
'data/nonrigid_z_processed.csv']

tri_filenames=['data_triangleconfig/pitch7_triangle_processed.csv', 'data_triangleconfig/x_triangle_processed.csv', 'data_triangleconfig/y_triangle_processed.csv',
'data_triangleconfig/z_triangle_processed.csv', 'data_triangleconfig/rollfinal_triangle_processed.csv',
'data_triangleconfig/yaw_triangle_processed.csv']


# dfx = pd.read_csv('data/data_tracker1_pitch.csv',sep=',')
dfx = pd.read_csv('data/nonrigid_y2_processed.csv',sep=',')
# dfx = pd.read_csv('RotationData/RotationalX.csv',sep=',')

dfx = dfx.dropna()


dfx.columns= ['y', 'x1', 'x2', 'x3','x4','x5','x6']
dfx=dfx[['x1','x2','x3','x4','x5','x6','y']]
dfx.head


######autocorrelation


x=dfx[["y"]]
#plot autocorrelation function
fig = tsaplots.plot_acf(x, lags=(len(dfx)-1))
plt.show()


###########add timestamp
N = len(dfx)


t = np.arange(0, N, 1).reshape(-1,1)
t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
t = np.array(np.round(t, 2))

print(t)

#############combine timestamp with RotationX dataframe


dataset = pd.DataFrame(np.concatenate((t, dfx), axis=1), 
                       columns=['t','x1', 'x2', 'x3','x4','x5','x6','y'])

deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]) for i in range(len(dataset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))

dataset.insert(1, '∆t', deltaT)


dataset.head(15)


############split dataset into training and testing sets

trainset = dataset.sample(frac=0.67, random_state=25)
testset = dataset.drop(trainset.index)
#trainset.head()
trainset.shape


############# linear model without window technique


gbr = GradientBoostingRegressor(random_state=0)

gbr.fit(trainset.iloc[:,:-1], trainset.iloc[:,-1])

t0 = time.time()
gbr_y = testset['y'].values
gbr_y_fit = gbr.predict(trainset.iloc[:,:-1])
gbr_y_pred = gbr.predict(testset.iloc[:,:-1])
tF = time.time()

gbr_residuals = gbr_y_pred - gbr_y
gbr_rmse = np.sqrt(np.sum(np.power(gbr_residuals,2)) / len(gbr_residuals))
print('RMSE = %.4f' % gbr_rmse)


# In[9]:


####### WindowSlider help function
ridge_rmse={}
ridge_rmse_list=[]
for m in range(100,(len(trainset) - 1),100):

    class WindowSlider(object):
        
        def __init__(self, window_size = m):        
            '''
            Window Slider object
            ====================
            w: window_size - number of time steps to look back
            o: offset between last reading and temperature
            r: response_size - number of time steps to predict
            l: maximum length to slide - (#observation - w)
            p: final predictors - (#predictors * w)
            '''
            self.w = window_size
            self.o = 0
            self.r = 1       
            self.l = 0
            self.p = 0
            self.names = []
        
        def re_init(self, arr):
            '''
            Helper function to initializate to 0 a vector
            '''
            arr = np.cumsum(arr)
            return arr - arr[0]
                

        def collect_windows(self, X, window_size=m, offset=0, previous_y=False):
            '''
            Input: X is the input matrix, each column is a variable
            Returns: diferent mappings window-output
            '''
            cols = len(list(X)) - 1
            N = len(X)
        
            self.o = offset
            self.w = window_size
            self.l = N - (self.w + self.r) + 1
            if not previous_y: self.p = cols * (self.w)
            if previous_y: self.p = (cols + 1) * (self.w)
        
            # Create the names of the variables in the window
            # Check first if we need to create that for the response itself
            if previous_y: x = cp.deepcopy(X)
            if not previous_y: x = X.drop(X.columns[-1], axis=1)  
        
            for j, col in enumerate(list(x)):        
                
                for i in range(self.w):
                
                    name = col + ('(%d)' % (i+1))
                    self.names.append(name)
        
            # Incorporate the timestamps where we want to predict
            for k in range(self.r):
            
                name = '∆t' + ('(%d)' % (self.w + k + 1))
                self.names.append(name)
            
            self.names.append('Y')
                
            df = pd.DataFrame(np.zeros(shape=(self.l, (self.p + self.r + 1))), 
                          columns=self.names)
        
            # Populate by rows in the new dataframe
            for i in range(self.l):
            
                slices = np.array([])
            
                # Flatten the lags of predictors
                for p in range(x.shape[1]):
            
                    line = X.values[i:self.w + i, p]
                    # Reinitialization at every window for ∆T
                    if p == 0: line = self.re_init(line)
                    
                    # Concatenate the lines in one slice    
                    slices = np.concatenate((slices, line)) 
 
                # Incorporate the timestamps where we want to predict
    
                line = np.array([self.re_init(X.values[i:i+self.w+self.r, 0])[-1]])
                y = np.array(X.values[self.w + i + self.r - 1, -1]).reshape(1,)
                slices = np.concatenate((slices, line, y))
            
                # Incorporate the slice to the cake (df)
                df.iloc[i,:] = slices
            
            return df
        
 
    w=m
    train_constructor = WindowSlider()
    train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

    test_constructor = WindowSlider()
    test_windows = test_constructor.collect_windows(dataset.iloc[(len(trainset))-w:,1:],
                                                previous_y=False)
#     test_windows = test_constructor.collect_windows(testset.iloc[len(dfx)-w:,1:],
#                                                 previous_y=False)
    
#     print(test_windows.shape)

    train_constructor_y_inc = WindowSlider()
    train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=True)

    test_constructor_y_inc = WindowSlider()
    test_windows_y_inc = test_constructor_y_inc.collect_windows(dataset.iloc[(len(trainset))-w:,1:],
                                                previous_y=True)
#     test_windows_y_inc = test_constructor_y_inc.collect_windows(testset.iloc[len(dfx)-w:,1:],
#                                                 previous_y=True)

    #train_windows.head(3)
    #test_windows.head(3)
    
    ridge_model = Ridge()
    ridge_model.fit(train_windows.iloc[:,:-1], train_windows.iloc[:,-1])



    t0 = time.time()
    ridge_y = test_windows['Y'].values
    ridge_y_fit = ridge_model.predict(train_windows.iloc[:,:-1])
    ridge_y_pred = ridge_model.predict(test_windows.iloc[:,:-1])
    tF = time.time()

    ridge_residuals = ridge_y_pred - ridge_y
    ridge_rmse[m] = np.sqrt(np.sum(np.power(ridge_residuals,2)) / len(ridge_residuals))
    ridge_rmse_list.append(ridge_rmse[m])
    
print(ridge_rmse_list)       


x_ridge=range(100,(len(trainset)-1),100)

y_ridge=ridge_rmse_list

plt.plot(x_ridge, y_ridge, label = "Ridge Regression")

  
# naming the x axis
plt.xlabel('Window Size')
# naming the y axis
plt.ylabel('RMSE')

  
# function to show the plot

plt.show()

#######GBR

####### WindowSlider help function
gbr_rmse={}
gbr_rmse_list=[]
for m in range(100,(len(trainset)-1),100):

    class WindowSlider(object):
        
        def __init__(self, window_size = m):        
            '''
            Window Slider object
            ====================
            w: window_size - number of time steps to look back
            o: offset between last reading and temperature
            r: response_size - number of time steps to predict
            l: maximum length to slide - (#observation - w)
            p: final predictors - (#predictors * w)
            '''
            self.w = window_size
            self.o = 0
            self.r = 1       
            self.l = 0
            self.p = 0
            self.names = []
        
        def re_init(self, arr):
            '''
            Helper function to initializate to 0 a vector
            '''
            arr = np.cumsum(arr)
            return arr - arr[0]
                

        def collect_windows(self, X, window_size=m, offset=0, previous_y=False):
            '''
            Input: X is the input matrix, each column is a variable
            Returns: diferent mappings window-output
            '''
            cols = len(list(X)) - 1
            N = len(X)
        
            self.o = offset
            self.w = window_size
            self.l = N - (self.w + self.r) + 1
            if not previous_y: self.p = cols * (self.w)
            if previous_y: self.p = (cols + 1) * (self.w)
        
            # Create the names of the variables in the window
            # Check first if we need to create that for the response itself
            if previous_y: x = cp.deepcopy(X)
            if not previous_y: x = X.drop(X.columns[-1], axis=1)  
        
            for j, col in enumerate(list(x)):        
                
                for i in range(self.w):
                
                    name = col + ('(%d)' % (i+1))
                    self.names.append(name)
        
            # Incorporate the timestamps where we want to predict
            for k in range(self.r):
            
                name = '∆t' + ('(%d)' % (self.w + k + 1))
                self.names.append(name)
            
            self.names.append('Y')
                
            df = pd.DataFrame(np.zeros(shape=(self.l, (self.p + self.r + 1))), 
                          columns=self.names)
        
            # Populate by rows in the new dataframe
            for i in range(self.l):
            
                slices = np.array([])
            
                # Flatten the lags of predictors
                for p in range(x.shape[1]):
            
                    line = X.values[i:self.w + i, p]
                    # Reinitialization at every window for ∆T
                    if p == 0: line = self.re_init(line)
                    
                    # Concatenate the lines in one slice    
                    slices = np.concatenate((slices, line)) 
 
                # Incorporate the timestamps where we want to predict
    
                line = np.array([self.re_init(X.values[i:i+self.w+self.r, 0])[-1]])
                y = np.array(X.values[self.w + i + self.r - 1, -1]).reshape(1,)
                slices = np.concatenate((slices, line, y))
            
                # Incorporate the slice to the cake (df)
                df.iloc[i,:] = slices
            
            return df
        
 
    w=m
    train_constructor = WindowSlider()
    train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

    test_constructor = WindowSlider()
    test_windows = test_constructor.collect_windows(dataset.iloc[len(trainset)-w:,1:],
                                                previous_y=False)

    train_constructor_y_inc = WindowSlider()
    train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=True)

    test_constructor_y_inc = WindowSlider()
    test_windows_y_inc = test_constructor_y_inc.collect_windows(dataset.iloc[len(trainset)-w:,1:],
                                                previous_y=True)

    #train_windows.head(3)
    #test_windows.head(3)
    
    gbr_model = GradientBoostingRegressor()
    gbr_model.fit(train_windows.iloc[:,:-1], train_windows.iloc[:,-1])



    t0 = time.time()
    gbr_y = test_windows['Y'].values
    gbr_y_fit = gbr_model.predict(train_windows.iloc[:,:-1])
    gbr_y_pred = gbr_model.predict(test_windows.iloc[:,:-1])
    tF = time.time()

    gbr_residuals = gbr_y_pred - gbr_y
    gbr_rmse[m] = np.sqrt(np.sum(np.power(gbr_residuals,2)) / len(gbr_residuals))
    gbr_rmse_list.append(gbr_rmse[m])
    print(gbr_rmse[m])
    
print(gbr_rmse_list)  

# x_gbr=range(100,(len(trainset)-1),100)

# y_gbr=gbr_rmse_list

# plt.plot(x_gbr, y_gbr, label = "Gradient Boosting Regressor")

  
# naming the x axis
plt.xlabel('Window Size')
# naming the y axis
plt.ylabel('RMSE')

  
# function to show the plot

plt.show()

######autocorrelation 

x=dfx[["y"]]
#plot autocorrelation function
fig = tsaplots.plot_acf(x, lags=(len(dfx)-1))
plt.show()


################# GBR
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0)

gbr.fit(train_windows.iloc[:,:-1], train_windows.iloc[:,-1])

t0 = time.time()
gbr_y = test_windows['Y'].values
gbr_y_fit = gbr.predict(train_windows.iloc[:,:-1])
gbr_y_pred = gbr.predict(test_windows.iloc[:,:-1])
tF = time.time()

gbr_residuals = gbr_y_pred - gbr_y
gbr_rmse = np.sqrt(np.sum(np.power(gbr_residuals,2)) / len(gbr_residuals))
print('RMSE = %.4f' % gbr_rmse)




