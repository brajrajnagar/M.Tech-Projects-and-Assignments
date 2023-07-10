import numpy as np
import pandas as pd
import sys


# Taking paths for test data directory
script_name, test_dir = sys.argv

# Values of Thetas as given in question
theta_given = np.array([3,1,2])

M = 1000000         # No. of Samples
# generating samples for x1, x1 and noise
x1 = np.random.normal(3, np.sqrt(4), M)
x2 = np.random.normal(-1, np.sqrt(4), M)
g_noise = np.random.normal(0, np.sqrt(2), M)

# Converting into 2D  array
x0 = np.array([np.ones(M)])
x1 = np.array([x1])
x2 = np.array([x2])
X_sample = np.concatenate((x0, x1, x2), axis=0)  #(3, 1000000)
Y_sample = np.matmul(theta_given, X_sample) + g_noise      #(1000000,)


# Assigning the batch size
batch_size = 1
splits = round(M/batch_size)
# Spliting into total No. batches
X_batch = np.array(np.array_split(X_sample, splits, axis=1))
Y_batch = np.array(np.array_split(Y_sample, splits))


# For training the model using stochastic gradient descent initializin Thetas 
Theta = np.zeros(3)
# Storing Thetas and cost with each update
Theta_store = np.array([Theta])
cost_store = np.array([])

eta = 0.001     #Learning Rate

# Calculating the cost
def cost_fn(X,Theta,Y,m):
    err = Y - np.matmul(Theta, X)
    J_theta = (1/(2*m)) * np.sum((err)**2)
    return J_theta
        

# Updating the Thetas
def grade_dec(X,Theta,Y,m):
    err2 = Y - np.matmul(Theta, X)
    sm = np.zeros(3)
    for j in range(m):
        sm += err2[j] * X[:,j]

    d_theta = (eta/m)*sm
    Theta += d_theta

# inital cost 
cost_store = np.append(cost_store, cost_fn(X_sample, Theta, Y_sample, Y_sample.size))

itr = 1
cond_sat = True
# Keep updating Thetas until stopping criteria hits
while(cond_sat):
    itr += 1
    for i in range(splits):
        grade_dec(X_batch[i], Theta, Y_batch[i], Y_batch[0].size)
        Theta_store = np.append(Theta_store, [Theta], axis=0)
        # print(Theta, itr, i)
        # for batch size 1 checkng convergence after 10000 samples
        # for batch size 100 checkng convergence after 100 samples
        # for batch size 10000 checkng convergence after 10 samples
        # for batch size 1000000 checkng convergence after each samples (i == 0)
        # if (i == 0):
        if (i%10000 == 0 and i != 0):
            temp_cost = cost_fn(X_sample, Theta_store[-2], Y_sample, Y_sample.size)
            cost_store = np.append(cost_store, temp_cost)
            temp_cost = cost_fn(X_sample, Theta, Y_sample, Y_sample.size)
            cost_store = np.append(cost_store, temp_cost)
            # print(Theta)
            if abs(cost_store[-2] - cost_store[-1]) <= 4e-4 :
                cond_sat = False
                break
            else:
                continue
    

# Model Training is  completed

dataset = np.array(pd.read_csv(test_dir+'/X.csv', header=None))
X_1 = np.array(dataset[:,0])
X_2 = np.array(dataset[:,1])

X0 = np.array([np.ones(X_1.shape)])
X1 = np.array([X_1])
X2 = np.array([X_2])
X = np.concatenate((X0, X1, X2), axis=0)
hy = np.matmul(Theta, X)

with open('result_2.txt', 'w+') as file_out:
    for item in hy:
        file_out.write(str(item))
        file_out.write("\n")