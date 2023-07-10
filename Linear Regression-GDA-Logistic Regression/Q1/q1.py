import numpy as np
import pandas as pd
import sys


# Taking paths for train data directory and test data directory
script_name, train_dir, test_dir = sys.argv
# Reading files with pandas library
lin_X = pd.read_csv(train_dir+f'/X.csv', header=None)
lin_Y = pd.read_csv(train_dir+f'/Y.csv', header=None)
# storing X,Y data into numpy array
X = np.array(lin_X).reshape(-1)
Y = np.array(lin_Y).reshape(-1)
m = X.size

# Data Normalization
X_mean = np.mean(X)
X_std = np.std(X)
for i in range(len(X)):
    X[i] = (X[i] - X_mean) / X_std

X = np.array([X])
X= np.append(np.ones(X.shape), X, axis=0)


# Initializing Theta
Theta = np.zeros(2)
# Storing Theta and cost values
Theta_store = np.array([Theta])
cost_store = np.array([])
#Learning Rate
eta = 0.025

# Calculating the cost
def cost_fn(X,Theta,Y):
    err = Y - np.matmul(Theta, X)
    J_theta = (1/(2*m)) * np.sum((err)**2)
    return J_theta
        
# Calculating the updated theta using gradient descent
def grade_dec(X,Theta,Y):
    err2 = Y - np.matmul(Theta, X)
    sm = np.zeros(2)
    for j in range(m):
        sm += err2[j] * X[:,j]
    
    d_theta = (eta/m)*sm
    Theta += d_theta

# inital cost 
cost_store = np.append(cost_store, cost_fn(X, Theta, Y))

cost_store_pre = 0
count = 0
# we keep updating theta values untill the stopping criteria hits
while (abs(cost_store[-1] - cost_store_pre) >= 1e-11):
    grade_dec(X, Theta, Y)
    Theta_store = np.append(Theta_store, [Theta], axis=0)
    cost_store_pre = cost_store[-1]
    temp_cost = cost_fn(X,Theta,Y)
    cost_store = np.append(cost_store, temp_cost)
    count += 1


test_X = pd.read_csv(test_dir+f'/X.csv', header=None)
Xt = np.array(test_X).reshape(-1)
# Data Normalization
Xt_mean = np.mean(Xt)
Xt_std = np.std(Xt)
for i in range(len(Xt)):
    Xt[i] = (Xt[i] - Xt_mean) / Xt_std
Xt = np.array([Xt])
Xt = np.append(np.ones(Xt.shape), Xt, axis=0)
hy = np.matmul(Theta, Xt)

with open('result_1.txt', 'w+') as file_out:
    for item in hy:
        file_out.write(str(item))
        file_out.write("\n")
