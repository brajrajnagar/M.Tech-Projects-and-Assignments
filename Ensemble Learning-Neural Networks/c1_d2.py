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


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train2,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.plot(ccp_alphas[::1000], impurities[::1000], marker="o", drawstyle="steps-post")
#the maximum effective alpha value is removed, because it's impurity is very high.
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig(dt_drug_review.out_path+'/2_c_d2_impVsalpha.jpg')

ccp_alphas = ccp_alphas[::1000]

clfs = []
count = 0
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train2, y_train)
    # print("clf.fit", count)
    count += 1
    clfs.append(clf)



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
plt.savefig(dt_drug_review.out_path+'/2_c_nodedepthVsalpha.jpg')



train_scores = [clf.score(X_train2, y_train) for clf in clfs]
test_scores = [clf.score(X_test2, y_test) for clf in clfs]
val_scores = [clf.score(X_val2, y_val) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.plot(ccp_alphas, val_scores, marker="o", label="validation", drawstyle="steps-post")
ax.legend()
plt.savefig(dt_drug_review.out_path+'/2_c_accVsalpha.jpg')


best_alpha = ccp_alphas[val_scores.index(max(val_scores))]


best_index = val_scores.index(max(val_scores))
with open(dt_drug_review.out_path+'/2_c.txt', 'w+') as file1:

    file1.write(f'train accuracy with respect to the best tree: {train_scores[best_index]*100}% \n')
    file1.write(f'test accuracy with respect to the best tree: {test_scores[best_index]*100}% \n')
    file1.write(f'validation accuracy with respect to the best tree: {max(val_scores*100)}% \n')