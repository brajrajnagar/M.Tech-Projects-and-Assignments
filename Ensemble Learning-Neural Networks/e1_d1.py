import pandas as pd
import dt_mammography

train_dataset = pd.read_csv(dt_mammography.train_path, index_col=False)

import numpy as np
train_df = train_dataset.replace(to_replace='?', value= np.nan)

X_train = train_df.iloc[:,1:5].copy()
y_train = train_df.iloc[:,5].copy()


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy='median')
X_train = pd.DataFrame(imp.fit_transform(X_train, y_train), columns=['Age', 'Shape', 'Margin', 'Density'])


test_dataset = pd.read_csv(dt_mammography.test_path, index_col=False)
test_df = test_dataset.replace(to_replace='?', value= np.nan)
X_test = test_df.iloc[:,1:5].copy()
y_test = test_df.iloc[:,5].copy()
X_test = pd.DataFrame(imp.fit_transform(X_test, y_test), columns=['Age', 'Shape', 'Margin', 'Density'])

val_dataset = pd.read_csv(dt_mammography.val_path, index_col=False)
val_df = val_dataset.replace(to_replace='?', value= np.nan)
X_val = val_df.iloc[:,1:5].copy()
y_val = val_df.iloc[:,5].copy()
X_val = pd.DataFrame(imp.fit_transform(X_val, y_val), columns=['Age', 'Shape', 'Margin', 'Density'])


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

from sklearn import tree
from matplotlib import pyplot as plt 
fig = plt.figure(figsize=(150,100))
_ = tree.plot_tree(clf, feature_names=X_train.columns,
                class_names={0:'benign', 1:'malignant'},
                filled=True, fontsize=12)
plt.savefig(dt_mammography.out_path+'/1e_DS1.jpg')

train_acc = clf.score(X_train, y_train)*100
test_acc = clf.score(X_test, y_test)*100
val_acc = clf.score(X_val, y_val)*100

with open(dt_mammography.out_path+'/1_e.txt', 'w+') as file1:
    file1.write(f'Reslts for Part (a): \n')
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')



# Part b

from sklearn.model_selection import GridSearchCV

param = {'max_depth':range(4,5),
            'min_samples_split':range(3,4), 'min_samples_leaf':range(3,4)}
grid = GridSearchCV(DecisionTreeClassifier(), param)
grid = grid.fit(X_train, y_train)


clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, min_samples_split=3)
clf = clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)


fig = plt.figure(figsize=(30,20))
_ = tree.plot_tree(clf, feature_names=X_train.columns,
                class_names={0:'benign', 1:'malignant'},
                filled=True, fontsize=12)
plt.savefig(dt_mammography.out_path+'/1e_b_DS1.jpg')

train_acc = clf.score(X_train, y_train)*100
test_acc = clf.score(X_test, y_test)*100
val_acc = clf.score(X_val, y_val)*100

with open(dt_mammography.out_path+'/1_e.txt', 'a') as file1:
    file1.write(f'Results for Part (b): \n')
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')


# Part c

clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
#the maximum effective alpha value is removed, because it's impurity is very high.
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig(dt_mammography.out_path+'/1e_c_Impurity_vs_effective_alpha.jpg')

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig(dt_mammography.out_path+'/1e_c_nodes_Depth_vs_alpha.jpg')


train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
val_scores = [clf.score(X_val, y_val) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
ax.legend()
plt.savefig(dt_mammography.out_path+'/1e_c_Accuracy_vs_alpha.jpg')


best_alpha = ccp_alphas[val_scores.index(max(val_scores))]
best_index = val_scores.index(max(val_scores))
with open(dt_mammography.out_path+'/1_e.txt', 'a') as file1:
    file1.write(f'Results for part (c): \n')
    file1.write(f'train accuracy with respect to the best tree: {train_scores[best_index]*100}% \n')
    file1.write(f'test accuracy with respect to the best tree: {test_scores[best_index]*100}% \n')
    file1.write(f'validation accuracy with respect to the best tree: {max(val_scores)*100}% \n')
    file1.write('\n')


best_clf = clfs[best_index]


fig = plt.figure(figsize=(30,20))
_ = tree.plot_tree(best_clf, feature_names=X_train.columns,
                class_names={0:'benign', 1:'malignant'},
                filled=True, fontsize=12)
plt.savefig(dt_mammography.out_path+'/1e_c_Dtree.jpg')




# patr d


from sklearn.ensemble import RandomForestClassifier

param = {'n_estimators': range(40,41),
            'max_features': range(4,5), 'min_samples_split':range(27,28), 'oob_score': [True]}
grid = GridSearchCV(RandomForestClassifier(), param)
grid = grid.fit(X_train, y_train)
oob_acc =  grid.best_score_*100


clf = RandomForestClassifier(max_features=4, min_samples_split=27, n_estimators=40)
clf = clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)*100
test_acc = clf.score(X_test, y_test)*100
val_acc = clf.score(X_val, y_val)*100


with open(dt_mammography.out_path+'/1_e.txt', 'a') as file1:
    file1.write(f'Results for Part (d): \n')
    file1.write(f'train accuracy: {train_acc}% \n')
    file1.write(f'out-of-bag accuracy: {oob_acc}% \n')
    file1.write(f'test accuracy: {test_acc}% \n')
    file1.write(f'validation accuracy: {val_acc}% \n')
    file1.write(f'\n')