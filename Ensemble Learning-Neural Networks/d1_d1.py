import pandas as pd
import dt_mammography

train_dataset = pd.read_csv(dt_mammography.train_path, index_col=False)

train_df = train_dataset.copy()
for col in train_dataset:
    train_df = train_df[train_df[col]!= '?']

X_train = train_df.iloc[:,1:5].copy()
y_train = train_df.iloc[:,5].copy()


test_dataset = pd.read_csv(dt_mammography.test_path, index_col=False)

test_df = test_dataset.copy()
for col in test_dataset:
    test_df = test_df[test_df[col]!= '?']

X_test = test_df.iloc[:,1:5].copy()
y_test = test_df.iloc[:,5].copy()

val_dataset = pd.read_csv(dt_mammography.val_path, index_col=False)

val_df = val_dataset.copy()
for col in val_dataset:
    val_df = val_df[val_df[col]!= '?']

X_val = val_df.iloc[:,1:5].copy()
y_val = val_df.iloc[:,5].copy()




from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()



from sklearn.model_selection import GridSearchCV

param = {'n_estimators': range(45,55),
            'max_features': range(1,len(X_train.columns)+1), 'min_samples_split':range(2,9), 'oob_score': [True]}
grid = GridSearchCV(RandomForestClassifier(), param)
grid = grid.fit(X_train, y_train)
oob_acc =  grid.best_score_






clf = RandomForestClassifier(max_features=1, min_samples_split=7, n_estimators=51)
clf = clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
val_acc = clf.score(X_val, y_val)

with open(dt_mammography.out_path+'/1_d.txt', 'w+') as file1:
    file1.write(f'train accuracy: {train_acc*100}% \n')
    file1.write(f'out-of-bag accuracy: {oob_acc*100}% \n')
    file1.write(f'test accuracy: {test_acc*100}% \n')
    file1.write(f'validation accuracy: {val_acc*100}% \n')