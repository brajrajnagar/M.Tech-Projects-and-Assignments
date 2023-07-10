import numpy as np
import pickle
import time
import sys
from sklearn.svm import SVC


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


def svm_liner_kernel(train_X, train_Y, kerneltype):
    SVC_linear = SVC(kernel = kerneltype)
    start = time.time()
    SVC_linear.fit(train_X,train_Y)
    end = time.time()
    t = end-start
    SV = SVC_linear.support_
    nSV = len(SVC_linear.support_)
    w = SVC_linear.coef_
    b = SVC_linear.intercept_.squeeze()
    acc = SVC_linear.score(test_X,test_Y)
    return t, SV, nSV, w, b, acc

def svm_gaussian_kernel(train_X, train_Y, kerneltype, g):
    SVC_gaussian = SVC(kernel = kerneltype, gamma = g)
    start = time.time()
    SVC_gaussian.fit(train_X,train_Y)
    end = time.time()
    t = end-start
    SV = SVC_gaussian.support_
    nSV = len(SVC_gaussian.support_)
    acc = SVC_gaussian.score(test_X,test_Y)
    return t, SV, nSV, acc


t_lin, SV_lin, nSV_lin, w, b, acc_lin = svm_liner_kernel(train_X, train_Y, 'linear')
t_gauss, SV_gauss, nSV_gauss, acc_gauss = svm_gaussian_kernel(train_X, train_Y, 'rbf', 'auto')



print(f'nSV for linear kernel: {nSV_lin}')
print(f'nSV for gaussian kernel: {nSV_gauss}')


print(f'w: {w}')
print(f'b: {b}')


print(f'Test set accuracy linear kernel: {acc_lin*100}%')
print(f'Test set accuracy gaussian kernel: {acc_gauss*100}%')


print(f'Training time for linear kernel: {t_lin}sec')
print(f'Training time for gaussian kernel: {t_gauss}sec')


with open('supportvectors_lin.txt', 'w+') as fp:
    for item in SV_lin:
        fp.write("%s\n" % item)


with open('supportvectors_gauss.txt', 'w+') as fp:
    for item in SV_gauss:
        fp.write("%s\n" % item)