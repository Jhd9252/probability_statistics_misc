# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:05:22 2022

@author: jhd9252
"""

import numpy as np
import pandas as pd
import math

# input: (jointProb: float, probA:float) -> float
    # assume proper input between [0,1], and jointProb <= p(A)
# process: P(B|A) = P(B intersection A) / P(A)
# output: return probability of P(B|A) rounded to 2 digits
    
def jointProbability(jointProb: float, probA: float) -> float:
    return round(jointProb / probA,2)

# input: 1D numpy array that contains arbitrary size of numbers
# process: compute the median of array
#          find difference of data point x - median
#          record the absolute value of the differences
#          compute the median of those abs values
# output: median absolute deviation of a 1D array

def medianAbsDeviation(arr):
    # return the median of absolute values of data points minus median.
    # or the median of absolute residuals from median
    return np.median([abs(x-np.median(arr)) for x in arr])

# input: (2d numpy array or pandas df, flag)
    # flag 1 = mean absolute error
    # flag 2 = root mean squared
    # flag 3 = cubic root mean error
    # etc. 
# process: run calculations according by flag 
# output: Return scalar number according to calculations by flag
# assumptions: valid arr input in form numpy ndarray or pandas dataframe, rows flexible according to first assumption.


def normalized_error(arr, flag):
    if isinstance(arr, np.ndarray):
        return (sum(abs(arr[:,0] - arr[:,1]) ** flag) / arr.shape[0]) ** (1/flag)
    elif isinstance(arr,pd.DataFrame):
        return (sum(abs(arr.iloc[:,0] - arr.iloc[:,1]) ** flag) / arr.shape[0]) ** (1/flag)     
    
# input: 2 args (data, confidence)
# process: find lower and upper bounds of sample data
#          extract data into arr, sort the arr
#          determine the lower and upper mass from second argument
#          determine nearest index in sample that corresponds to probability masses
#          extract the values from that index as the upper and lower bounds
# output: lower and upper bounds as 2 variables, or a tuple
# assumption 1: first arg can be anything, with default = 1d np.ndarray
# assumption 2: second arg is a real number between [0.01, 99.99] as %
def findBounds(data, confidence):
    writtenBy: 'Jonathan Dinh'
    arr = []
    # grab data - default is 1d ndarray
    if isinstance(data, np.ndarray):
        arr = [x for x in data[:,0]]
    # grab data - list 1d
    elif isinstance(data, list):
        arr = data
    # grab data - dataframe 1d
    elif isinstance(data, pd.DataFrame):
        arr = [x for x in data.iloc[:,0]]
    # grab data - string csv 1d
    elif isinstance(data, str):
        if '.csv' in data:
            df = pd.read_csv(data)
            arr = [x for x in df.iloc[:,0]]
            
    arr = sorted(arr)
    
    # find prob mass in tail and head
    lower_mass = (100 - confidence) / 2
    upper_mass = 100 - lower_mass

    # from probability masses, find the nearest index in sample that matches
    # ensure int obviously
    lower_index = round(len(arr)/100 * lower_mass)-1 
    upper_index = round(len(arr)/100 * upper_mass)-1 

    # grab those values at the corresponding indexes 
    lowerBound = arr[lower_index] 
    upperBound = arr[upper_index] 
    
    # return as tuple
    return (lowerBound, upperBound)

"""
Purpose: 
    Most classic desciptive statistics calculate over entire range of variable.
    These approaches assumes that the generative process is stationary or constant.
    
    When exploring data such as time series or spacial locations, assumption is not true. 
    It is of interest to see how the correlation develops over time, or within a window. 
    
    We want a function that computes parameters over a window, and then be able to move said window. 
    
    Input: 
        1. data set, either a 1d or 2d numpy array
        2. flag for which parameter to calculate, 1 = mean, 2 = SD, 3 = correlation
        # note that for std, sample or population std is valid for this assignment. 
        3. window size or the subset of how many numbers of dataset to compute 
           
    Process: 
        1. calculate parameter from input2, over input3. 
        2. continously shift window by 1 to the right and calculate. Repeat. 
        3. These results should be appnede to output array. One result per calculation. 
    Return:
        1. result array, should be less length than input array
    Assumptions:
        1. 2d input if flag is correlation
        2. 1d input if flag is mean or SD
        3. assume good input, window less than or equal input length
              
"""
import numpy as np
import pandas as pd
import math

def SlidingDescriptives(dataset, flag, window):
    writtenBy = 'Jonathan Dinh'
    arr = []
    pos = 0
    # first check the flag to determine the dimensions of dataset
    if flag == 1: # then we are calculating the mean with 1d numpy array
        while pos + window <= len(dataset):
            # get window and calculate and append and move pos + 1
            arr.append(np.mean([x for x in dataset[pos:window+pos]]))
            pos += 1
            
    elif flag == 2: # then we are calculating SD with 1d input numpay array
        while pos + window <= len(dataset):
            arr.append(np.std([x for x in dataset[pos:window+pos]])) # population
            pos += 1
    elif flag == 3: # then we are calculating correlation
        while pos + window <= dataset.shape[0]:
            arr.append(np.corrcoef(dataset[pos:pos+window,0],dataset[pos:pos+window,1])[0][1])
            pos += 1
    return arr
