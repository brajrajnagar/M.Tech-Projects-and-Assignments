import pandas as pd
import dt_mammography


train_dataset = pd.read_csv(dt_mammography.train_path, index_col=False)

train_df = train_dataset.copy()
for col in train_dataset:
    train_df = train_df[train_df[col]!= '?']


X_train = train_df.iloc[:,1:5].copy()
y_train = train_df.iloc[:,5].copy()




from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


from sklearn import tree
from matplotlib import pyplot as plt 
fig = plt.figure(figsize=(150,100))
_ = tree.plot_tree(clf, feature_names=X_train.columns,
                class_names={0:'benign', 1:'malignant'},
                filled=True, fontsize=12)
plt.savefig(dt_mammography.out_path+'/1a_DS1.jpg')




pred_train = clf.predict(X_train)

from sklearn.metrics import accuracy_score
with open(dt_mammography.out_path+'/1_a.txt', 'w+') as file1:
    file1.write(f"training accuracy: {accuracy_score(y_train, pred_train)*100}% \n")







test_dataset = pd.read_csv(dt_mammography.test_path, index_col=False)

test_df = test_dataset.copy()
for col in test_dataset:
    test_df = test_df[test_df[col]!= '?']

X_test = test_df.iloc[:,1:5].copy()
y_test = test_df.iloc[:,5].copy()

pred_test = clf.predict(X_test)
with open(dt_mammography.out_path+'/1_a.txt', 'a') as file1:
    file1.write(f"test accuracy: {accuracy_score(y_test, pred_test)*100}% \n")



val_dataset = pd.read_csv(dt_mammography.val_path, index_col=False)

val_df = val_dataset.copy()
for col in val_dataset:
    val_df = val_df[val_df[col]!= '?']

X_val = val_df.iloc[:,1:5].copy()
y_val = val_df.iloc[:,5].copy()

pred_val = clf.predict(X_val)
with open(dt_mammography.out_path+'/1_a.txt', 'a') as file1:
    file1.write(f"validation accuracy: {accuracy_score(y_val, pred_val)*100}% \n")





