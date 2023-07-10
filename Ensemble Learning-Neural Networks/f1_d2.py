import pandas as pd
import numpy as np
import dt_drug_review

train_dataset = pd.read_csv(dt_drug_review.train_path)
test_dataset = pd.read_csv(dt_drug_review.test_path)
val_dataset = pd.read_csv(dt_drug_review.val_path)


X_train = train_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_train = train_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

X_val = val_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_val = val_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')


from sklearn.feature_extraction.text import TfidfVectorizer
cv1 = TfidfVectorizer(stop_words='english')
X_train_rev = cv1.fit_transform(X_train['review'])
X_test_rev = cv1.transform(X_test['review'])
X_val_rev = cv1.transform(X_val['review'])

X_train_cond = cv1.fit_transform(X_train['condition'])
X_test_cond = cv1.transform(X_test['condition'])
X_val_cond = cv1.transform(X_val['condition'])


import scipy
X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])
X_val2 = scipy.sparse.hstack([X_val_cond, X_val_rev])



from lightgbm import LGBMClassifier
clf = LGBMClassifier()

from sklearn.model_selection import GridSearchCV

param = {'n_estimators':range(50,500,50)}
grid = GridSearchCV(LGBMClassifier(), param)
grid = grid.fit(X_train2, y_train)
best_param = grid.best_params_


clf = LGBMClassifier(n_estimators=best_param['n_estimators'])
clf.fit(X_train2,y_train)
train_acc = clf.score(X_train2, y_train)
test_acc = clf.score(X_test2, y_test)
val_acc = clf.score(X_val2, y_val)


with open(dt_drug_review.out_path+'/2_e.txt', 'w+') as file1:
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')