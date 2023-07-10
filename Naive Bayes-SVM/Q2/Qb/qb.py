import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers
import sys
from cvxopt import matrix
import pickle
import time


script_name, train_path, test_path = sys.argv


train_file = train_path+'/train_data.pickle'
test_file = test_path+'/test_data.pickle'


def data_build(data_file):
    X = np.array([])
    Y = np.array([])
    with open(data_file, 'rb') as file:
        data=pickle.load(file)
        m,n = 0, np.array(data['data']).shape[1]*np.array(data['data']).shape[2]*np.array(data['data']).shape[3]
        for i in range(len(data['labels'])):
            if (data['labels'][i]==2):
                data_temp = (data['data'][i].astype('double').flatten())/255
                X = np.append(X, data_temp)
                Y = np.append(Y, 1)
                m += 1
            if (data['labels'][i]==3):
                data_temp = (data['data'][i].astype('double').flatten())/255
                X = np.append(X, data_temp)
                Y = np.append(Y, -1)
                m += 1
        X = X.reshape(m,n)
        return X, Y



train_X,train_Y = data_build(train_file)
test_X,test_Y = data_build(test_file)


def gauss_kernel(x,z,g=0.001):
    ker = np.exp(-g*np.dot(x-z,x-z))
    return ker


start = time.time()
m = len(train_X) #4000
n = len(train_Y) #4000
YT = np.zeros((n,n), 'double')
np.fill_diagonal(YT, train_Y)
P1 = np.zeros((m,m), 'double')
for i in range(len(train_X)):
	for j in range(len(train_X)):
		P1[i][j] = gauss_kernel(train_X[i],train_X[j])
P = matrix(np.dot(YT,np.dot(P1,YT)))
q = matrix(np.full((m,1),-1,dtype='double'))
G1 = np.zeros((m, m), 'double')
np.fill_diagonal(G1, 1) # daigonal matrix of 1 (4000,4000)
G2 = np.zeros((m, m), 'double')
np.fill_diagonal(G2, -1)
G = matrix(np.concatenate([G1,G2],axis=0))
h1 =  np.full((m,1),1,dtype='double')
h2 = np.zeros((m,1),dtype='double')
h = matrix(np.concatenate([h1,h2],axis=0))
A = matrix(train_Y.astype('double').reshape(1,-1))
b = matrix(np.zeros((1,1),dtype='double'))

sloution = solvers.qp(P,q,G,h,A,b)
alphas = np.array(sloution['x'])
end = time.time()



S_V = []
for i in range(len(train_X)):
    if alphas[i]>1e-6 and alphas[i]<999999e-6:
        S_V.append(i)

b = 0
for i in range(len(train_X)):
        b_temp = 0
        for j in range(len(train_X)):
            b_temp += (alphas[j]*train_Y[j]*gauss_kernel(train_X[i],train_X[j])).squeeze()
        b += train_Y[i]-b_temp
b=b/len(train_X)



Y_pred = []
for i in range(len(test_X)):
    w_x = 0
    for j in range(len(train_X)):
        w_x += (alphas[j]*train_Y[j]*gauss_kernel(test_X[i],train_X[j])).squeeze()
    y = w_x+b
    Y_pred.append(1) if y>0 else Y_pred.append(-1)

ac = np.sum(Y_pred==test_Y)/len(Y_pred)*100

print(f'training time for model: {end-start}sec')
print(f'support vectors : {len(S_V)}')
print(f'percentage of training samples constitute the support vectors: {len(S_V)/len(train_X)*100}%')
print(f'Test set accuracy: {ac}%')




al =  alphas.reshape(-1)
dict = {}
for i in range(len(alphas)):
    dict[i] = al[i]

sorted_value_index = np.argsort(dict.values())
dictionary_keys = list(dict.keys())
sorted_dict = {dictionary_keys[i]: sorted(
    dict.values())[i] for i in range(len(dictionary_keys))}

dict_keys = list(sorted_dict.keys())
top_keys = list(reversed(dict_keys[len(dict_keys)-5:len(dict_keys)]))

fig1 = plt.figure(figsize=(10,2))
for i in range(5):
	ax = fig1.add_subplot(1,5,i+1)
	ax.imshow(train_X[top_keys[i]].reshape(32, 32,3))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.savefig('top5.png')


with open('supportvectors.txt', 'w+') as fp:
    for item in S_V:
        fp.write("%s\n" % item)