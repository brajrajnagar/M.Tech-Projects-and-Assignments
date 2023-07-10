import numpy as np
import pandas as pd
import sys


# Taking paths for train data directory and test data directory
script_name, train_dir, test_dir = sys.argv

X_dataset = pd.read_csv(train_dir+"/X.csv", header=None)
Y_dataset = pd.read_csv(train_dir+"/Y.csv", header=None)
X = np.array(X_dataset)
X1 = np.array(X[:,0])
X2 = np.array(X[:,1])
Y = np.array(Y_dataset)


# Data Normalization
X1 = (X1 - np.mean(X1)) / np.std(X1)
X2 = (X2 - np.mean(X2)) / np.std(X2)

x1 = np.array([X1])
x2 = np.array([X2])
X = np.concatenate((x1.T, x2.T), axis=1)
dataset = np.concatenate((X, Y), axis=1)


canada_ = []
alaska_ = []

for i in range(Y.size):
    if Y[i,0] == 'Canada':
        canada_.append(np.array([X1[i],X2[i]]))
    elif Y[i,0] == 'Alaska':
        alaska_.append(np.array([X1[i],X2[i]]))

canada_ = np.array(canada_)
alaska_ = np.array(alaska_)


# Alaska = 0    canada=1
phi = len(canada_)/len(Y)
mu = np.array([ np.mean(alaska_, axis=0), np.mean(canada_, axis=0)])
diff1 = alaska_ - mu[0]
diff2 = canada_ - mu[1]
temp = np.concatenate((diff1, diff2))
cov_mat = np.matmul(np.transpose(temp),temp)/len(Y)


"""
the equation of the boundary separating the two regions
        mx + c = 0
"""
mu_diffT = np.transpose(mu[0] - mu[1])
cov_mat_inv = np.linalg.inv(cov_mat)
m = np.matmul(mu_diffT,cov_mat_inv)

p1 = np.matmul(np.transpose(mu[0]), np.matmul(cov_mat_inv,mu[0]))
p2 = np.matmul(np.transpose(mu[1]), np.matmul(cov_mat_inv,mu[1]))
p3 = np.log((1-phi)/phi)
c = (-p1 + p2 + p3)/2

Te = np.append(c, m)


# Reading Test datasets
X_dataset = pd.read_csv(test_dir+"/X.csv", header=None)
X = np.array(X_dataset)
X1 = np.array(X[:,0])
X2 = np.array(X[:,1])
# Test Data Normalization
X1 = (X1 - np.mean(X1)) / np.std(X1)
X2 = (X2 - np.mean(X2)) / np.std(X2)

M = X1.size
# Converting into 2D  array
x0 = np.array([np.ones(M)])
x1 = np.array([X1])
x2 = np.array([X2])

X_norm = np.concatenate((x0, x1, x2), axis=0)
n = np.matmul(Te, X_norm)
hy = (1/(1+ np.exp(n)))

for id in range(len(hy)):
    if hy[id] > 0.5:
        hy[id] = 1
    else:
        hy[id] = 0

with open('result_4.txt', 'w+') as file_out:
    for item in hy:
        if item == 1:
            file_out.write('Canada')
            file_out.write("\n")
        elif item == 0:
            file_out.write('Alaska')
            file_out.write("\n")
