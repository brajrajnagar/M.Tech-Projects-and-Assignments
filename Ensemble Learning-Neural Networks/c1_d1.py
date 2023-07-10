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
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
#the maximum effective alpha value is removed, because it's impurity is very high.
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig(dt_mammography.out_path+'/1c_impVsalpha.jpg')



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
plt.savefig(dt_mammography.out_path+'/1c_nodedepthVsalpha.jpg')



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
plt.savefig(dt_mammography.out_path+'/1c_accVsalpha.jpg')




best_alpha = ccp_alphas[val_scores.index(max(val_scores))]


best_index = val_scores.index(max(val_scores))
with open(dt_mammography.out_path+'/1_c.txt', 'w+') as file1:
    file1.write(f'train accuracy with respect to the best tree: {train_scores[best_index]*100}% \n')
    file1.write(f'test accuracy with respect to the best tree: {test_scores[best_index]*100}% \n')
    file1.write(f'validation accuracy with respect to the best tree: {max(val_scores)*100}% \n')



best_clf = clfs[best_index]

from sklearn import tree
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(best_clf, feature_names=X_train.columns,
                class_names={0:'benign', 1:'malignant'},
                filled=True, fontsize=12)
plt.savefig(dt_mammography.out_path+'/1c_DS1_DT.jpg')