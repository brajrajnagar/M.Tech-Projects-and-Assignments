import pandas as pd
import numpy as np
import time
import dt_drug_review

train_dataset = pd.read_csv(dt_drug_review.train_path)
test_dataset = pd.read_csv(dt_drug_review.test_path)
val_dataset = pd.read_csv(dt_drug_review.val_path)


n_samples =  [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]


datasets = []
for i in n_samples:
    datasets.append(train_dataset.sample(n=i, replace=True))


##### Part (a)

test_accuracies = []
train_times = []

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

for spl in range(len(n_samples)):
    X_train = datasets[spl].iloc[:, 0:2].replace(to_replace=np.nan, value='')
    y_train = datasets[spl].iloc[:, 2].replace(to_replace=np.nan, value='')

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()

    cv1 = TfidfVectorizer(stop_words='english')
    X_train_rev = cv1.fit_transform(X_train['review'])
    X_test_rev = cv1.transform(X_test['review'])

    X_train_cond = cv1.fit_transform(X_train['condition'])
    X_test_cond = cv1.transform(X_test['condition'])

    import scipy
    X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
    X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    start = time.time()
    clf = clf.fit(X_train2, y_train)
    end =  time.time()
    train_times.append(end-start)

    test_accuracies.append(clf.score(X_test2, y_test))


for i in range(len(test_accuracies)):
    test_accuracies[i] =  test_accuracies[i]*100
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, test_accuracies, label='Tets accuracies')
plt.xlabel('number of examples')
plt.ylabel('Test Accuracy in %')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1ga_accuracies.jpg')

# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, train_times, label='Train times')
plt.xlabel('number of examples')
plt.ylabel('Time in sec.')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1ga_time.jpg')


### Part (b)

test_accuracies = []
train_times = []

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

for spl in range(len(n_samples)):
    X_train = datasets[spl].iloc[:, 0:2].replace(to_replace=np.nan, value='')
    y_train = datasets[spl].iloc[:, 2].replace(to_replace=np.nan, value='')

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()

    cv1 = TfidfVectorizer(stop_words='english')
    X_train_rev = cv1.fit_transform(X_train['review'])
    X_test_rev = cv1.transform(X_test['review'])

    X_train_cond = cv1.fit_transform(X_train['condition'])
    X_test_cond = cv1.transform(X_test['condition'])

    import scipy
    X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
    X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV

    param = {'max_depth':range(20,50,10),
                'min_samples_split':range(2,4), 'min_samples_leaf':range(1,3)}
    grid = HalvingGridSearchCV(DecisionTreeClassifier(), param)
    grid = grid.fit(X_train2, y_train)
    best_param = grid.best_params_

    clf = DecisionTreeClassifier(max_depth = best_param['max_depth'], min_samples_leaf = best_param['min_samples_leaf'], min_samples_split = best_param['min_samples_split'])
    # clf = DecisionTreeClassifier(max_depth=20, min_samples_split=2, min_samples_leaf=2)
    start =  time.time()
    clf = clf.fit(X_train2, y_train)
    end = time.time()
    train_times.append(end-start)
    test_accuracies.append(clf.score(X_test2, y_test))

for i in range(len(test_accuracies)):
    test_accuracies[i] =  test_accuracies[i]*100
# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, test_accuracies, label='Tets accuracies')
plt.xlabel('number of examples')
plt.ylabel('Test Accuracy in %')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gb_accuracies.jpg')


# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, train_times, label='Train times')
plt.xlabel('number of examples')
plt.ylabel('Time in sec.')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gb_time.jpg')



### Part (d)

test_accuracies = []
train_times = []

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

for spl in range(len(n_samples)):
    X_train = datasets[spl].iloc[:, 0:2].replace(to_replace=np.nan, value='')
    y_train = datasets[spl].iloc[:, 2].replace(to_replace=np.nan, value='')

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()

    cv1 = TfidfVectorizer(stop_words='english')
    X_train_rev = cv1.fit_transform(X_train['review'])
    X_test_rev = cv1.transform(X_test['review'])

    X_train_cond = cv1.fit_transform(X_train['condition'])
    X_test_cond = cv1.transform(X_test['condition'])

    import scipy
    X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
    X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()

    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    param = {'n_estimators': range(50,500,50),
                'max_features': [0.4,0.5,0.6,0.7,0.8], 'min_samples_split':range(2,12,2), 'oob_score': [True]}
    grid = HalvingGridSearchCV(RandomForestClassifier(), param)
    grid = grid.fit(X_train2, y_train)
    oob_acc =  grid.best_score_*100
    best_param = grid.best_params_


    clf = RandomForestClassifier(max_features=best_param['max_features'], min_samples_split=['min_samples_split'], n_estimators=['n_estimators'])
    # clf = RandomForestClassifier(max_features=0.5, n_estimators=100, min_samples_split=3, oob_score=True)
    start =  time.time()
    clf = clf.fit(X_train2, y_train)
    end = time.time()
    train_times.append(end-start)
    test_accuracies.append(clf.score(X_test2, y_test))


for i in range(len(test_accuracies)):
    test_accuracies[i] =  test_accuracies[i]*100
# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, test_accuracies, label='Tets accuracies')
plt.xlabel('number of examples')
plt.ylabel('Test Accuracy in %')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gd_accuracies.jpg')


# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, train_times, label='Train times')
plt.xlabel('number of examples')
plt.ylabel('Time in sec.')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gd_time.jpg')



#### Part (e)

test_accuracies = []
train_times = []

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')
y_test = le.fit_transform(y_test)

for spl in range(len(n_samples)):
    X_train = train_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
    y_train = train_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')
    y_train = le.fit_transform(y_train)

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()

    cv1 = TfidfVectorizer(stop_words='english')
    X_train_rev = cv1.fit_transform(X_train['review'])
    X_test_rev = cv1.transform(X_test['review'])

    X_train_cond = cv1.fit_transform(X_train['condition'])
    X_test_cond = cv1.transform(X_test['condition'])

    import scipy
    X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
    X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])

    from xgboost import XGBClassifier
    clf = XGBClassifier()

    from sklearn.model_selection import GridSearchCV

    param = {'max_depth':range(40,80,10),
                'n_estimators':range(50,500,50), 'subsample': [0.4,0.5,0.6,0.7,0.8]}
    grid = GridSearchCV(XGBClassifier(), param)
    grid = grid.fit(X_train2, y_train)
    best_param = grid.best_params_

    clf = XGBClassifier(max_depth=best_param['max_depth'], n_estimators=best_param['n_estimators'], subsample=best_param['subsample'])
    start =  time.time()
    clf = clf.fit(X_train2, y_train)
    end = time.time()
    train_times.append(end-start)
    test_accuracies.append(clf.score(X_test2, y_test))

for i in range(len(test_accuracies)):
    test_accuracies[i] =  test_accuracies[i]*100
# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, test_accuracies, label='Tets accuracies')
plt.xlabel('number of examples')
plt.ylabel('Test Accuracy in %')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1ge_accuracies.jpg')

# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, train_times, label='Train times')
plt.xlabel('number of examples')
plt.ylabel('Time in sec.')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1ge_time.jpg')




### Part (f)

test_accuracies = []
train_times = []

X_test = test_dataset.iloc[:, 0:2].replace(to_replace=np.nan, value='')
y_test = test_dataset.iloc[:, 2].replace(to_replace=np.nan, value='')

for spl in range(len(n_samples)):
    X_train = datasets[spl].iloc[:, 0:2].replace(to_replace=np.nan, value='')
    y_train = datasets[spl].iloc[:, 2].replace(to_replace=np.nan, value='')

    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()

    cv1 = TfidfVectorizer(stop_words='english')
    X_train_rev = cv1.fit_transform(X_train['review'])
    X_test_rev = cv1.transform(X_test['review'])

    X_train_cond = cv1.fit_transform(X_train['condition'])
    X_test_cond = cv1.transform(X_test['condition'])

    import scipy
    X_train2 = scipy.sparse.hstack([X_train_cond, X_train_rev])
    X_test2 = scipy.sparse.hstack([X_test_cond, X_test_rev])

    from lightgbm import LGBMClassifier
    clf = LGBMClassifier()

    from sklearn.model_selection import GridSearchCV
    param = {'n_estimators':range(50,500,50)}
    grid = GridSearchCV(LGBMClassifier(), param)
    grid = grid.fit(X_train2, y_train)
    best_param = grid.best_params_

    clf = LGBMClassifier(n_estimators = best_param['n_estimators'])

    start =  time.time()
    clf = clf.fit(X_train2, y_train)
    end = time.time()
    train_times.append(end-start)
    test_accuracies.append(clf.score(X_test2, y_test))


for i in range(len(test_accuracies)):
    test_accuracies[i] =  test_accuracies[i]*100
# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, test_accuracies, label='Tets accuracies')
plt.xlabel('number of examples')
plt.ylabel('Test Accuracy in %')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gf_accuracies.jpg')

# from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(n_samples, train_times, label='Train times')
plt.xlabel('number of examples')
plt.ylabel('Time in sec.')
plt.legend()
plt.savefig(dt_drug_review.out_path+'/1gf_time.jpg')