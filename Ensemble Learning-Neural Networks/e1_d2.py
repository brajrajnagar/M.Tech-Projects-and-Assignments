import pandas as pd
import numpy as np
import dt_drug_review
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


train_dataset = pd.read_csv(dt_drug_review.train_path)
test_dataset = pd.read_csv(dt_drug_review.test_path)
val_dataset = pd.read_csv(dt_drug_review.val_path)


X_train = train_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_train = train_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')
y_train = le.fit_transform(y_train)

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')
y_test = le.fit_transform(y_test)

X_val = val_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_val = val_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')
y_val = le.fit_transform(y_val)



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


from xgboost import XGBClassifier
clf = XGBClassifier()


from sklearn.model_selection import GridSearchCV

param = {'max_depth':range(40,80,10),
            'n_estimators':range(50,500,50), 'subsample': [0.4,0.5,0.6,0.7,0.8]}
grid = GridSearchCV(XGBClassifier(), param)
grid = grid.fit(X_train2, y_train)
best_param = grid.best_params_



clf = XGBClassifier(max_depth=best_param['max_depth'], n_estimators=best_param['n_estimators'], subsample=best_param['subsample'])
clf.fit(X_train2,y_train)
train_acc = clf.score(X_train2, y_train)
test_acc = clf.score(X_test2, y_test)
val_acc = clf.score(X_val2, y_val)


with open(dt_drug_review.out_path+'/2_e.txt', 'w+') as file1:
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')