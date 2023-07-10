import numpy as np
import pandas as pd
import sys

# Taking paths for train data directory and test data directory
script_name, train_dir, test_dir = sys.argv
# Reading files with pandas library
X_dataset = pd.read_csv(train_dir+"/X.csv", header=None)
Y_dataset = pd.read_csv(train_dir+"/Y.csv", header=None)
X = np.array(X_dataset)
X1 = np.array(X[:,0])
X2 = np.array(X[:,1])
Y = np.array(Y_dataset).reshape(-1)

# Data Normalization
X1 = (X1 - np.mean(X1)) / np.std(X1)
X2 = (X2 - np.mean(X2)) / np.std(X2)

M = X1.size
# Converting into 2D  array
x0 = np.array([np.ones(M)])
x1 = np.array([X1])
x2 = np.array([X2])

X_norm = np.concatenate((x0, x1, x2), axis=0)


# Initaializing Thetas
Theta = np.zeros(3)

# Maximum log likelihood calculation
def MLL(X,Theta,Y,m):
    n = np.matmul(Theta, X)  #1by100
    h = (1/(1+ np.exp(-n)))    #1by100
    L_theta = 0
    for i in range(m):
        L_theta += (Y[i]*np.log(h[i]) + (1-Y[i])*np.log(1-h[i]))

    return L_theta

# Storng log likelihood values
LL = np.array([MLL(X_norm, Theta, Y, Y.size)])

# Optimizing log liklihood using newton's method
def newopt(X,Theta,Y,m):
    n = np.matmul(Theta, X)  #1by100
    h = (1/(1+ np.exp(-n)))    #1by100
    err2 = Y - h
    d_theta = np.zeros(3)
    for j in range(m):
        d_theta += err2[j] * X[:,j]

    I = np.identity(Y.size)
    temp = I * np.dot(h,np.transpose(1-h))
    Hes = np.matmul(X, np.matmul(temp, np.transpose(X)))
    Theta += np.matmul(np.linalg.inv(Hes),np.transpose(d_theta))
    return Theta, d_theta
    

cond = True
itr = 1
# Keep updating Thetas using optimization until log likihood function maximize
while(cond):
    Theta, d_Theta =  newopt(X_norm, Theta, Y, Y.size)
    L = MLL(X_norm, Theta, Y, Y.size)
    LL = np.append(LL, L)
    if abs(LL[-2]-LL[-1]) <= 0.0001:
        cond = False
    else:
        itr += 1

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
n = np.matmul(Theta, X_norm)
hy = (1/(1+ np.exp(-n)))

for id in range(len(hy)):
    if hy[id] > 0.5:
        hy[id] = 1
    else:
        hy[id] = 0

with open('result_3.txt', 'w+') as file_out:
    for item in hy:
        file_out.write(str(item))
        file_out.write("\n")