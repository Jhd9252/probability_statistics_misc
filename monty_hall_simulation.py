# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:33:03 2022
@author: jhd9252
https://mathworld.wolfram.com/MontyHallProblem.html
"""

# Monty Hall Problem
# Problem states the following;
# There is a room with 3 doors
# Behind 2 of the 3 doors, there is a goat (no prize)
# Behind the 3rd door is a brand new car (prize)
# If you pick a door, the host (who knows what's behind all the doors)
# opens on of the other 2 doors, revealing a goat (no prize)
# You then have a choice of switching doors, or staying with your initial choice. 

# The correct answer is that you do switch doors.
 
# Choice 1: Is to stay with the initial door. This choice has a 1/3 chance of 
# winning the car. No matter what you picked, and the actions taken by the host 
# afterwords, if you do not switch doors, you have a 1/3 chance of winning. There
# are no changes to the probability of winning. 

# Choice 2: Is to switch doors after the host opens a door you didn't choose. 
# With this new information, that the door the host opened is without the prize,
# the last door must have a 2/3 chance of having the prize. 

# Essentially, choosing the initial door has a 1/3 chance of winning
# Then by partitioning, the other two doors must have a combined 2/3 chance of winning
# Therefore if the host opens the second door containing a good, by switching, the last door
# contains 2/3 chance of winning. 
# Probability(Staying with same door or Partition A) = 1/3 
# Probabilitiy(Switching doors into partition B) with one of the doors showing no prize = 2/3 chance

# This statistical illusion stems from the fact that the outcomes are not random
# The actions taken by the host is not random
# And the probabilities change during the course of actions. 

# Initially, all the doors had a 1/3 chance of winning
# However by using insider knowledge, and opening a door without the prize,
# the probabilities of switching 


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import statistics

# set number of trials and number of doors
doors = 3
trials = 1000

# helper function to pick a door (for determining prize and initial pick)
def pick(): return random.choice([x for x in range(1, doors+1)])

# function that does the simulation 1 time
# Always switching doors
def trial():
    # randomly determine the door with prize
    prize = pick()
    # randomly choose initial door
    initial = pick()
    # the host will open a door without the prize
    # cannot be initial door
    # cannot be door with prize
    host = random.choice([x for x in [1,2,3] if x not in [prize, initial]])
    # always switch doors -> Choose door that is not initial, or host
    switched = random.choice([x for x in [1,2,3] if x not in [initial, host]])
    # return true or false
    return switched == prize

# function that will 
# 1. return an array of results of multiple simulations
# 2. return count of True
# 3. return count of False
def trial_multiple(num_trials):
    arr = []
    for i in range(num_trials):
        arr.append(trial())
    return arr, arr.count(True), arr.count(False)


# lets run the simulation trial amount of times.
results, success, failure = trial_multiple(trials)

# now let's plot the counts of successes and failures 
fig = plt.figure(figsize = (10,10))
plt.bar(['Failure', 'Success'], [failure, success])
plt.xlabel('Results After Switching Doors')
plt.ylabel('Number of Trials')

# Now let's try plotting the distribution of results
# with sample size (previously trials)
# with n samples
# In other words, we will run the simulation 1000 times, in batches in 1000. 
sample = 100
n = 1000

# We will obtain the mean success rate of each sample
# number of successes / number of total trials per n.
# we should have a means arr of len (n)
means = []
for i in range(n):
    sample_results = trial_multiple(sample)[0]
    mean = np.mean(sample_results)
    means.append(mean)
    
fig2 = plt.figure(figsize = (10,10))



# plot our observed distribution
numBins = 100
plt.hist(means, numBins)  
plt.xlabel('Average Probabilities of Success After Switching')
plt.ylabel('Count')

# now we will fit a normal distribution according to our observed data to 
# see that the averages of each sample is centered around the prediction of 2/3 chance
# fit the normal distribution to data
means = sorted(means)
mu, std = norm.fit(means)

# plot the probability density function of normal distribution
xmin, xmax = min(means), max(means)
x = np.linspace(xmin, xmax, 100)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax)
p = norm.pdf(x,mu,std)

# scale and plot the normal distribution 
# to do this, multiply distribution by number of records, divided by number of bins
plt.plot(x, p*n/numBins, 'k', linewidth = 3)
plt.grid()

# plot the mean
plt.plot([np.mean(means), np.mean(means)], [0,400],color='red',linewidth=5)
plt.show()



# From the first graph, where we ran a single trial 1000 times,
# we can see that when switching doors, the probability of success is around 660/100 or 2/3.
# While the probability of failure over the long term is 330/1000 or 1/3
# This falls in line with the expected predictions.

# From the second graph, we ran 100 trials, 1000 times. We obtained the mean 
# of each batch of 100 trials, 1000 times. We then plotted the frequency of each 
# mean. In addition, we fitted a standard normal distribution to scale over the histogram. 
# From this we can see that we can expect the highest concentration of successes around
# the 0.66 probability (switching doors). A normal distribution has an E[X_n] = mu, 
# with E[avg(X_n)] = mu, which is 0.66. This falls in line with our expectations. 

    

    

    

