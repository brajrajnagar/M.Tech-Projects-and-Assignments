import pandas as pd
import dt_mammography

train_dataset = pd.read_csv(dt_mammography.train_path, index_col=False)
test_dataset = pd.read_csv(dt_mammography.test_path, index_col=False)
val_dataset = pd.read_csv(dt_mammography.val_path, index_col=False)

import numpy as np
train_df = train_dataset.replace(to_replace='?', value= np.nan)
test_df = test_dataset.replace(to_replace='?', value= np.nan)
val_df = val_dataset.replace(to_replace='?', value= np.nan)

X_train = np.array(train_df.iloc[:,1:5].copy())
y_train = np.array(train_df.iloc[:,5].copy())

X_test = np.array(test_df.iloc[:,1:5].copy())
y_test = np.array(test_df.iloc[:,5].copy())

X_val = np.array(val_df.iloc[:,1:5].copy())
y_val = np.array(val_df.iloc[:,5].copy())


from xgboost import XGBClassifier
clf = XGBClassifier()


from sklearn.model_selection import GridSearchCV

param = {'max_depth':range(4,10,1),
            'n_estimators':range(10,50,10), 'subsample': [i/10 for i in range(1,6,1)]}
grid = GridSearchCV(XGBClassifier(), param)
grid = grid.fit(X_train, y_train)
best_param = grid.best_params_

clf = XGBClassifier(max_depth= best_param['max_depth'], n_estimators= best_param['n_estimators'], subsample= best_param['subsample'])
clf.fit(X_train,y_train)
train_acc = clf.score(X_train, y_train)*100
test_acc = clf.score(X_test, y_test)*100
val_acc = clf.score(X_val, y_val)*100

with open(dt_mammography.out_path+'/1_f.txt', 'w+') as file1:
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')