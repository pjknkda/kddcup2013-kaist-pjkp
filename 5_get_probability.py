import collections
import csv
import itertools
import logging
import numpy as np
import os
import sys
import pickle as serializer

from config import *


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


from models import *

from features import *

trains = readCSV(TRAIN_FILE)


def build_feature_db():
    X = []
    Y = []

    for i, (aid, ok_pids_str, no_pids_str) in enumerate(trains):
        if i % 100 == 0:
            print('{}: {},'.format(i, aid))

        ok_pids = set(map(int, ok_pids_str.split()))
        no_pids = set(map(int, no_pids_str.split()))

        both_pids = ok_pids & no_pids

        ok_pids -= both_pids
        no_pids -= both_pids

        for pid in ok_pids:
            X.append(make_feature_vector(aid, pid))
            Y.append(1)

        for pid in no_pids:
            X.append(make_feature_vector(aid, pid))
            Y.append(-1)

    X = np.array(X)
    Y = np.array(Y)

    with open(FEATURE_DB_FILE, 'wb') as f:
        serializer.dump((X, Y), f, protocol=-1)


if not os.path.isfile(FEATURE_DB_FILE):
    build_feature_db()

with open(FEATURE_DB_FILE, 'rb') as f:
    X, Y = serializer.load(f)


def get_feature_with_missing_value(X):
    nan_feature_mean = []

    for feature_idx in range(X.shape[1]):
        nan_checker = np.isnan(X[:, feature_idx])
        if 0 < nan_checker.sum():
            # replace nan to avg.
            feature_mean = np.mean(X[np.nonzero(~nan_checker), feature_idx])
            X[np.nonzero(nan_checker), feature_idx] = feature_mean
            nan_feature_mean.append((feature_idx, feature_mean))

    return nan_feature_mean

feature_with_missing_value = get_feature_with_missing_value(X)

print('feature mean: ', np.mean(X, axis=0))
print('feature std: ', np.std(X, axis=0))


# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=3000)

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10000,
#                                     verbose=1)

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators=1000,
                                        verbose=1)


classifier.fit(X, Y)

print(classifier.score(X, Y))

try:
    for i, extractor_name in enumerate(extractor_names):
        print('{:25s}: {}'.format(extractor_name, classifier.feature_importances_[i]))
except:
    pass


def draw_decision_graph():
    print(X)
    import matplotlib.pyplot as plt

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    samples = 10000

    indexes = np.random.permutation(len(X))[:samples]

    X_small = X[indexes, :]
    Y_small = Y[indexes]

    #X_small = normalize(X_small, norm='l2', axis=0)

    #pca = TruncatedSVD(n_components=2)
    #X_reduced = pca.fit_transform(X_small)
    X_reduced = X_small[:, :2]

    ax = plt.subplot()
    ax.set_title("Predictions")
    #ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y_small, s=50)

    plt.tight_layout()
    plt.show()

# draw_decision_graph()

queries = []

with open(TEST_FILE, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for row in reader:
        aid = int(row[0])
        pids = np.array(list(map(int, row[1].split())))
        queries.append((aid, pids))

use_sample_db = False
sample_db = dict()


if os.path.isfile(SAMPLE_DB_FILE):
    with open(SAMPLE_DB_FILE, 'rb') as f:
        use_sample_db = True
        sample_db = serializer.load(f)


def sort_papers(aid, pids):
    sample_X = []

    pids = np.unique(pids)

    if use_sample_db:
        sample_X = sample_db[aid]
    else:
        for pid in pids:
            sample_X.append(make_feature_vector(aid, pid, missing_value_info=feature_with_missing_value))
        sample_db[aid] = sample_X

    ok_idx = classifier.classes_.tolist().index(1)

    prob_Y = classifier.predict_proba(sample_X)

    return pids[np.argsort(prob_Y[:, ok_idx])[::-1]]


with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    f.write('AuthorId,PaperIds\n')
    for i, (aid, pids) in enumerate(queries):
        if i % 100 == 0:
            print('{}: {},'.format(i, aid))

        new_pids = sort_papers(aid, pids)
        f.write('{},{}\n'.format(
            aid,
            ' '.join(
                map(str, new_pids)
            )
        ))

with open(SAMPLE_DB_FILE, 'wb') as f:
    serializer.dump(sample_db, f, protocol=-1)
