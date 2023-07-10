import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as met
import neural



train = pd.read_csv(neural.train_path,header = None, index_col=False).to_numpy()
test = pd.read_csv(neural.test_path,header = None, index_col=False).to_numpy()



X_train = train[:,:-1]/255
y_train = train[:,-1]

X_test = test[:,:-1]/255
y_test = test[:,-1]



from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), activation="relu", solver='sgd',max_iter=85,
                        learning_rate_init=0.1)
start = time.time()
clf = clf.fit(X_train,y_train)
end = time.time()

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)


with open(neural.out_path+"/g.txt", "w+") as file1:
    file1.write(f'train accuracy: {train_acc*100}% \n')
    file1.write(f'test accuracy: {test_acc*100}% \n')
    file1.write(f'training time: {end-start} sec \n')