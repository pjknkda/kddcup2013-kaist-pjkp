import collections
import csv
import itertools
import logging
import numpy as np
import os
import pickle as serializer
import sys
import time

from config import *


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


from models import *

from features import *

trains = readCSV(TRAIN_FILE)
trains_refined = []

# remove invalid pid which lies on both ok_pid and no_pid
for aid, ok_pids_str, no_pids_str in trains:
    ok_pids = set(map(int, ok_pids_str.split()))
    no_pids = set(map(int, no_pids_str.split()))

    both_pids = ok_pids & no_pids

    ok_pids -= both_pids
    no_pids -= both_pids

    trains_refined.append((aid, ok_pids, no_pids))


Y = []

for aid, ok_pids, no_pids in trains_refined:
    for pid in ok_pids:
        Y.append(1)
    for pid in no_pids:
        Y.append(0)

print('Feature extration [Train]')
os.makedirs(TRAIN_FEATURE_DB_FOLDER, mode=0o644, exist_ok=True)

train_features = dict()
missing_value_info = dict()

for feature_name in feature_names:
    print('- {:25s}: '.format(feature_name), end=" ")

    feature_dump_location = TRAIN_FEATURE_DB_FOLDER + '/{}.dump'.format(feature_name)

    feature_X = []

    if os.path.isfile(feature_dump_location):
        with open(feature_dump_location, 'rb') as f:
            feature_X = serializer.load(f)
        print('cached')
    else:
        starting_time = time.time()

        extractor = get_feature_extractor(feature_name)

        for aid, ok_pids, no_pids in trains_refined:
            for pid in ok_pids:
                feature_X.append(extractor(aid, pid))
            for pid in no_pids:
                feature_X.append(extractor(aid, pid))

        feature_X = np.array(feature_X)

        with open(feature_dump_location, 'wb') as f:
            serializer.dump(feature_X, f, protocol=-1)

        print('takes {:.4f}s'.format(time.time() - starting_time))

    train_features[feature_name] = feature_X


X = []

for i in range(len(Y)):
    x = []

    for feature_name in feature_names:
        x.append(train_features[feature_name][i])

    X.append(x)

X = np.array(X)

from sklearn.preprocessing import Imputer
missing_value_imputer = Imputer(missing_values=np.nan,
                                strategy='mean',
                                axis=0)
X = missing_value_imputer.fit_transform(X)

print('Feature statistics [Train]')

feature_mean = np.mean(X, axis=0)
feature_std = np.std(X, axis=0)

for i, feature_name in enumerate(feature_names):
    print('- {:25s}: mean {}, std {}'.format(feature_name, feature_mean[i], feature_std[i]))


def make_prediction_target(filepath):
    # Read test file
    queries = []

    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            aid = int(row[0])
            pids = np.array(list(map(int, row[1].split())))
            unique_pids = np.unique(pids)
            pids_queue = set(unique_pids)
            remain_pids = []
            for pid in pids:
                if pid in pids_queue:
                    pids_queue.remove(pid)
                else:
                    remain_pids.append(pid)
            queries.append((aid, unique_pids, remain_pids))

    print('Feature extration [{}]'.format(filepath))

    target_feature_db_folder = TARGET_FEATURE_DB_FOLDER + filepath.split('/')[-1]
    os.makedirs(target_feature_db_folder, mode=0o644, exist_ok=True)

    test_features = dict()

    for feature_name in feature_names:
        print('- {:25s}: '.format(feature_name), end=" ")

        feature_dump_location = target_feature_db_folder + '/{}.dump'.format(feature_name)

        feature_X = []

        if os.path.isfile(feature_dump_location):
            with open(feature_dump_location, 'rb') as f:
                feature_X = serializer.load(f)
            print('cached')
        else:
            starting_time = time.time()

            feature_mean = np.mean(train_features[feature_name])
            extractor = get_feature_extractor(feature_name)

            for aid, pids, remain_pids in queries:
                for pid in pids:
                    feature_X.append(extractor(aid, pid))

            feature_X = np.array(feature_X)

            with open(feature_dump_location, 'wb') as f:
                serializer.dump(feature_X, f, protocol=-1)

            print('takes {:.4f}s'.format(time.time() - starting_time))

        test_features[feature_name] = feature_X

    test_X = []
    test_x_idx = 0
    for aid, pids, remain_pids in queries:
        for pid in pids:
            x = []

            for feature_name in feature_names:
                x.append(test_features[feature_name][test_x_idx])

            test_X.append(x)
            test_x_idx += 1

    test_X = np.array(test_X)

    test_X = missing_value_imputer.transform(test_X)

    return queries, test_X

test_queries, test_X = make_prediction_target(TEST_FILE)

# Classifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def PlotDecisionArea():
    import matplotlib.pyplot as plt

    plot_colors = "rb"
    cmap = plt.cm.RdBu
    plot_step = 0.1  # fine step width for decision surface contours
    RANDOM_SEED = 13  # fix the seed on each iteration
    n_classes = 2

    nX = np.array(X)

    # We only take the two corresponding features
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_classes)
    y = np.array(Y)
    nX = pca.fit_transform(nX)
    nqX = pca.fit_transform(test_X)

    # Shuffle
    idx = np.arange(nX.shape[0])
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idx)
    nX = nX[idx[:1000]]
    y = y[idx[:1000]]

    idx = np.arange(nqX.shape[0])
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idx)
    nqX = nqX[idx[:1000]]

    # Standardize
    mean = nX.mean(axis=0)
    std = nX.std(axis=0)
    nX = (nX - mean) / std
    nqX = (nqX - mean) / std

    # Train
    model = ExtraTreesClassifier(n_estimators=500)
    clf = model.fit(nX, y)

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    x_min, x_max = nX[:, 0].min() - 1, nX[:, 0].max() + 1
    y_min, y_max = nX[:, 1].min() - 1, nX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Choose alpha blend level with respect to the number of estimators
    # that are in use (noting that AdaBoost can use fewer estimators
    # than its maximum if it achieves a good enough fit early on)
    estimator_alpha = 1.0 / len(clf.estimators_)
    for tree in clf.estimators_:
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

    # Plot the training points, these are clustered together and have a
    # black outline
    for i, c in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(nX[idx, 0], nX[idx, 1], c=c, label='class ' + str(i),
                    cmap=cmap)

    # Plot our quries
    for nqx in nqX:
        plt.scatter(nqx[0], nqx[1], c='y', label='query')

    plt.suptitle("Classifiers on data subsets [2d PCA on features]")
    plt.axis("tight")

    plt.show()

classifier = GradientBoostingClassifier(n_estimators=1000,
                                        verbose=1)

# Training classifier

# PlotDecisionArea()

print('Training classifier')
classifier.fit(X, Y)


print('Mean accuracy: {}'.format(classifier.score(X, Y)))

print('Feature importances')

try:
    for i, feature_name in enumerate(feature_names):
        print('{:25s}: {:5f}'.format(feature_name, classifier.feature_importances_[i]))
except:
    print(classifier.feature_importances_)

# Prediction
print('Prediction on test data')
predict_Y = classifier.predict_proba(test_X)


# make output file
print('Print results')

ok_idx = classifier.classes_.tolist().index(1)

with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    f.write('AuthorId,PaperIds\n')

    predict_y_idx = 0
    for aid, pids, remain_pids in test_queries:
        probabilites = []
        for pid in pids:
            probabilites.append(predict_Y[predict_y_idx][ok_idx])
            predict_y_idx += 1

        sorted_pids = pids[np.argsort(probabilites)[::-1]]
        f.write('{},{}\n'.format(
            aid,
            ' '.join(
                map(str, list(sorted_pids))
            )
        ))


from score import *


def valid_score_along_iteration():
    import matplotlib.pyplot as plt

    valid_solution = readCSV(VALID_SOLUTION_FILE)
    valid_pids = []
    for aid, pids_str, _ in valid_solution:
        valid_pids.append(list(map(int, pids_str.split())))

    valid_quries, valid_X = make_prediction_target(VALID_FILE)

    test_score = np.zeros(int(classifier.get_params()['n_estimators'] / 10), dtype=np.float64)
    for i, pred_y in enumerate(classifier.staged_predict_proba(valid_X)):
        if i % 10 != 0:
            continue

        staged_solution = []
        predict_y_idx = 0
        for aid, pids, remain_pids in valid_quries:
            probabilites = []
            for pid in pids:
                probabilites.append(pred_y[predict_y_idx][ok_idx])
                predict_y_idx += 1

            sorted_pids = pids[np.argsort(probabilites)[::-1]]
            staged_solution.append(sorted_pids)

        test_score[int(i / 10)] = mapk(valid_pids, staged_solution)

    top_10_score_indexes = np.argsort(test_score)[::-1][:10]
    print('Top 10 score indexes: ', top_10_score_indexes)
    print('Top 10 scores', test_score[top_10_score_indexes])

    plt.plot((np.arange(test_score.shape[0]) + 1), test_score,
             label='score')

    plt.legend(loc='lower right')
    plt.xlabel('Boosting Iterations (10x)')
    plt.ylabel('Valid Set Score')

    plt.show()

valid_score_along_iteration()
