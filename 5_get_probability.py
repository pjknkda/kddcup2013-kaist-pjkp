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
        Y.append(-1)

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

        # replace nan value to mean value
        nan_indexes = np.isnan(feature_X)
        if np.any(nan_indexes):
            feature_mean = np.mean(feature_X[np.nonzero(~nan_indexes)])
            feature_X[np.nonzero(nan_indexes)] = feature_mean

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


print('feature mean: ', np.mean(X, axis=0))
print('feature std: ', np.std(X, axis=0))


# Read test file
queries = []

with open(TEST_FILE, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for row in reader:
        aid = int(row[0])
        pids = np.array(list(map(int, row[1].split())))
        queries.append((aid, pids))


print('Feature extration [Test]')
os.makedirs(TEST_FEATURE_DB_FOLDER, mode=0o644, exist_ok=True)

test_features = dict()

for feature_name in feature_names:
    print('- {:25s}: '.format(feature_name), end=" ")

    feature_dump_location = TEST_FEATURE_DB_FOLDER + '/{}.dump'.format(feature_name)

    feature_X = []

    if os.path.isfile(feature_dump_location):
        with open(feature_dump_location, 'rb') as f:
            feature_X = serializer.load(f)
        print('cached')
    else:
        starting_time = time.time()

        feature_mean = np.mean(train_features[feature_name])
        extractor = get_feature_extractor(feature_name, feature_mean)

        for aid, pids in queries:
            for pid in pids:
                feature_X.append(extractor(aid, pid))

        feature_X = np.array(feature_X)

        with open(feature_dump_location, 'wb') as f:
            serializer.dump(feature_X, f, protocol=-1)

        print('takes {:.4f}s'.format(time.time() - starting_time))

    test_features[feature_name] = feature_X

test_X = []
test_x_idx = 0
for aid, pids in queries:
    for pid in pids:
        x = []

        for feature_name in feature_names:
            x.append(train_features[feature_name][test_x_idx])

        test_X.append(x)
        test_x_idx += 1


# Classifier

print('Training classifier')

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

print('Mean accuracy: {}'.format(classifier.score(X, Y)))
print('Feature importances')

for i, feature_name in enumerate(feature_names):
    print('{:25s}: {}'.format(feature_name, classifier.feature_importances_[i]))


# Prediction
print('Prediction on test data')
predict_Y = classifier.predict_proba(test_X)


# make output file
print('Print results')

ok_idx = classifier.classes_.tolist().index(1)

with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    f.write('AuthorId,PaperIds\n')

    predict_y_idx = 0
    for aid, pids in queries:
        probabilites = []
        for pid in pids:
            probabilites.append(predict_Y[predict_y_idx][ok_idx])
            predict_y_idx += 1

        sorted_pids = pids[np.argsort(probabilites)[::-1]]
        f.write('{},{}\n'.format(
            aid,
            ' '.join(
                map(str, sorted_pids)
            )
        ))
