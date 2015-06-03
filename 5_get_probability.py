import collections
import csv
import itertools
import logging
import numpy as np
import os
import sys
import pickle as serializer

# Dataset
TEST_FILE = '../data/Test.csv'
TRAIN_FILE = '../data/Train.csv'
RESULT_FILE = './result.csv'

# Meta data
META_DB_FILE = './meta_db.dump'
CONFERENCE_FILE = '../data/Conference.csv'
JOURNAL_FILE = '../data/Journal.csv'
PAPER_FILE = '../data/Paper.csv'
AUTHOR_FILE = './NormalizeAuthor.csv'
PAPER_AUTHOR_FILE = './NormalizePaperAuthor.csv'

# Cache
FEATURE_DB_FILE = './feature_db.dump'
SAMPLE_DB_FILE = './sample_db.dump'


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def readCSV(filepath):
    def row_generator():
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            fieldnames = next(reader)
            int_field_idxes = []
            for i in range(len(fieldnames)):
                if fieldnames[i].endswith('Id'):
                    int_field_idxes.append(i)

            for row in reader:
                for int_field_idx in int_field_idxes:
                    row[int_field_idx] = int(row[int_field_idx])
                yield row

        print('read complete : {}'.format(filepath))

    return row_generator()


class Authors(object):
    IDX_NAME = 0
    IDX_AFF = 1

    def __init__(self, _data=None):
        if _data is not None:
            self.rows = _data
            return

        self.rows = dict(
            ((row[0], tuple(row[1:])) for row in readCSV(AUTHOR_FILE))
        )

    @property
    def data(self):
        return self.rows

    def get(self, aid):
        return self.rows.get(aid)


class Publications(object):
    IDX_SHORT_NAME = 0
    IDX_FULL_NAME = 1
    IDX_URL = 2
    IDX_ORIGINAL_ID = 3

    def __init__(self, _data=None):
        if _data is not None:
            self.rows, self.inv_mapping = _data
            return

        conference_rows = readCSV(CONFERENCE_FILE)
        journal_rows = readCSV(JOURNAL_FILE)

        self.rows = dict()
        self.inv_mapping = dict()

        pub_id = 0
        for row in conference_rows:
            pub_id += 1
            self.inv_mapping[row[0]] = pub_id
            self.rows[pub_id] = tuple(row[1:] + [row[0]])

        for row in journal_rows:
            pub_id += 1
            self.inv_mapping[-row[0]] = pub_id
            self.rows[pub_id] = tuple(row[1:] + [-row[0]])

    @property
    def data(self):
        return (self.rows, self.inv_mapping)

    def get(self, pub_id):
        return self.rows.get(pub_id)

    def inv_map(self, conference_id, journal_id):
        if conference_id is not None:
            return self.inv_mapping.get(conference_id)
        else:
            return self.inv_mapping.get(-journal_id)


class Papers(object):
    IDX_TITLE = 0
    IDX_YEAR = 1
    IDX_PUB_ID = 2
    IDX_KEYWORDS = 3

    def __init__(self, publications=None, _data=None):
        if _data is not None:
            self.rows = _data
            return

        def _convert_row_JK_MANSE(row):
            return tuple([
                row[1],
                int(row[2]),
                publications.inv_map(row[3], row[4]),
                row[5].split()
            ])
        self.rows = dict(
            ((row[0], _convert_row_JK_MANSE(row)) for row in readCSV(PAPER_FILE))
        )

    @property
    def data(self):
        return self.rows

    def get(self, pid):
        return self.rows.get(pid)


class PaperAuthors(object):
    IDX_NAME = 0
    IDX_AFF = 1

    def __init__(self, _data=None):
        if _data is not None:
            self.rows, self.pids, self.aids = _data
            return

        self.rows = dict()
        self.pids = collections.defaultdict(list)
        self.aids = collections.defaultdict(list)

        for row in readCSV(PAPER_AUTHOR_FILE):
            self.rows[(row[0], row[1])] = tuple(row[2:])
            self.pids[row[0]].append((row[0], row[1]))
            self.aids[row[1]].append((row[0], row[1]))

    @property
    def data(self):
        return (self.rows, self.pids, self.aids)

    def get(self, pid, aid):
        return self.rows.get((pid, aid))

    def get_by_pid(self, pid):
        return self.pids.get(pid, [])

    def get_by_aid(self, aid):
        return self.aids.get(aid, [])


def build_metadata_db():
    authors = Authors()

    publications = Publications()

    papers = Papers(publications=publications)

    paper_authors = PaperAuthors()

    with open(META_DB_FILE, 'wb') as f:
        serializer.dump((authors.data, publications.data, papers.data, paper_authors.data), f, protocol=-1)


if not os.path.isfile(META_DB_FILE):
    build_metadata_db()


if (os.path.isfile(FEATURE_DB_FILE)
        and os.path.isfile(SAMPLE_DB_FILE)):
    print('No need to load metadata')
else:
    with open(META_DB_FILE, 'rb') as f:
        authors_data, publications_data, papers_data, paper_authors_data = serializer.load(f)
        authors = Authors(_data=authors_data)
        publications = Publications(_data=publications_data)
        papers = Papers(_data=papers_data)
        paper_authors = PaperAuthors(_data=paper_authors_data)

    print('Loading metadata is completed.')


if (os.path.isfile(FEATURE_DB_FILE)
        and os.path.isfile(SAMPLE_DB_FILE)):
    print('No need to load feature extractor')
else:
    def BayesAuthorToPaper():
        CONDENSE_DB_FILE = './condense_db.dump'
        INDEX_FILE = './index.dump'

        with open(CONDENSE_DB_FILE, 'rb') as f:
            condense_db = serializer.load(f)

        authors, paper_authors, name_id_mapping = condense_db

        def build_joint_table():
            # aid to nids (in paper_authors)
            np.matrix.sort(paper_authors, order=['aid', 'nid', 'pid'])

            aid_to_nids = dict()
            old_aid = -1
            for pid, aid, nid in paper_authors:
                if old_aid != aid:
                    old_aid = aid
                    aid_to_nids[aid] = []
                aid_to_nids[aid].append(nid)

            # aid to nids (in authors)
            np.matrix.sort(authors, order=['aid', 'nid'])

            old_aid = -1
            for aid, nid in authors:
                if old_aid != aid:
                    old_aid = aid
                    if aid not in aid_to_nids:
                        aid_to_nids[aid] = []
                aid_to_nids[aid].append(nid)

            # nid to pids (in paper_authors)
            np.matrix.sort(paper_authors, order=['nid', 'pid', 'aid'])

            nid_to_pids = dict()
            old_nid = -1
            for pid, aid, nid in paper_authors:
                if old_nid != nid:
                    old_nid = nid
                    nid_to_pids[nid] = []
                nid_to_pids[nid].append(pid)

            aid_to_nids = dict(((k, np.array(v)) for k, v in aid_to_nids.items()))
            nid_to_pids = dict(((k, np.array(v)) for k, v in nid_to_pids.items()))

            with open(INDEX_FILE, 'wb') as f:
                serializer.dump((aid_to_nids, nid_to_pids), f, protocol=-1)

        if not os.path.isfile(INDEX_FILE):
            build_joint_table()

        with open(INDEX_FILE, 'rb') as f:
            aid_to_nids, nid_to_pids = serializer.load(f)

        def calculator(aid, pid):
            nids_arr = aid_to_nids[aid]
            unique_nids = np.unique(nids_arr)

            prob = 0

            for nid in unique_nids:
                pids_given_nid_arr = nid_to_pids[nid]
                p1 = np.sum(pids_given_nid_arr == pid) / len(pids_given_nid_arr)
                p2 = 1.0 * np.sum(nids_arr == nid) / len(nids_arr)

                prob += p1 * p2

            return prob

        return calculator

    bayes_aid_to_pid = BayesAuthorToPaper()

    def AuthorNameDiffer():
        from fuzzywuzzy import fuzz

        def calculator(aid, pid):
            a_row = authors.get(aid)
            pa_row = paper_authors.get(pid, aid)

            if a_row is None or pa_row is None:
                return np.nan

            if (a_row[Authors.IDX_NAME] == '' or
                    pa_row[PaperAuthors.IDX_NAME]) == '':
                return np.nan

            sim = fuzz.ratio(
                a_row[Authors.IDX_NAME],
                pa_row[PaperAuthors.IDX_NAME]
            )
            return sim

        return calculator

    author_name_differ = AuthorNameDiffer()

    def AffiliationNameDiffer():
        from fuzzywuzzy import fuzz

        def calculator(aid, pid):
            a_row = authors.get(aid)
            pa_row = paper_authors.get(pid, aid)

            if a_row is None or pa_row is None:
                return np.nan

            if (a_row[Authors.IDX_AFF] == '' or
                    pa_row[PaperAuthors.IDX_AFF]) == '':
                return np.nan

            sim = fuzz.ratio(
                a_row[Authors.IDX_AFF],
                pa_row[PaperAuthors.IDX_AFF]
            )
            return sim

        return calculator

    affiliation_name_differ = AffiliationNameDiffer()

    def PaperYear():
        def calculator(pid):
            p_row = papers.get(pid)

            if p_row is None:
                return np.nan

            if not (1500 <= p_row[Papers.IDX_YEAR] <= 2013):  # filter invalid info
                return np.nan

            return p_row[Papers.IDX_YEAR]

        return calculator

    paper_year = PaperYear()

    def AuthorNumPaper():
        def calculator(aid):
            pid_aid_list = paper_authors.get_by_aid(aid)
            return len(pid_aid_list)

        return calculator

    author_num_paper = AuthorNumPaper()

    def AuthorNumCoauthor():
        def calculator(aid, pid):
            target_paper_authors = set()
            for ipid, iaid in paper_authors.get_by_pid(pid):
                target_paper_authors.add(iaid)

            author_coauthors = set()
            for ipid, iaid in paper_authors.get_by_aid(aid):
                for copid, coaid in paper_authors.get_by_pid(ipid):
                    author_coauthors.add(coaid)

            return len(target_paper_authors & author_coauthors)

        return calculator

    author_num_coauthor = AuthorNumCoauthor()

    def AuthorNumPublication():
        def calculator(aid):
            my_publications = set()
            for ipid, iaid in paper_authors.get_by_aid(aid):
                paper = papers.get(ipid)
                if paper is None:
                    continue
                if paper[Papers.IDX_PUB_ID] == 0:
                    continue
                my_publications.add(paper[Papers.IDX_PUB_ID])

            return len(my_publications)

        return calculator

    author_num_publication = AuthorNumPublication()

    def PaperTopicDist():
        TOPIC_FILE = './lda_topic.dump'

        with open(TOPIC_FILE, 'rb') as f:
            num_topics, topic_result = serializer.load(f)

        def calculator(pid):
            paper = papers.get(pid)
            if paper is None:
                return np.array([np.nan] * num_topics)
            if paper[Papers.IDX_PUB_ID] is None:
                return np.array([np.nan] * num_topics)

            publication = publications.get(paper[Papers.IDX_PUB_ID])

            return topic_result[publication[Publications.IDX_ORIGINAL_ID]]

        return calculator

    paper_topic_dist = PaperTopicDist()

    def AuthrPaperTopicSim():
        TOPIC_FILE = './lda_topic.dump'

        with open(TOPIC_FILE, 'rb') as f:
            num_topics, topic_result = serializer.load(f)

        def calculator(aid, pid):
            pass

        return calculator

    author_paper_topic_sim = AuthrPaperTopicSim()

    def make_feature_vector(aid, pid, normalize_info=None):
        feature_vec = np.array([
            bayes_aid_to_pid(aid, pid),
            author_name_differ(aid, pid),
            affiliation_name_differ(aid, pid),
            paper_year(pid),
            author_num_paper(aid),
            author_num_coauthor(aid, pid),
            author_num_publication(aid),
        ])
        feature_vec = np.concatenate([feature_vec, paper_topic_dist(pid)])

        if normalize_info is not None:
            for feature_idx, feature_mean in normalize_info:
                if np.isnan(feature_vec[feature_idx]):
                    feature_vec[feature_idx] = feature_mean

        return feature_vec

trains = readCSV(TRAIN_FILE)


def build_feature_db():
    X = []
    Y = []

    for aid, ok_pids_str, no_pids_str in trains:
        print(aid)

        ok_pids = list(map(int, ok_pids_str.split()))
        no_pids = list(map(int, no_pids_str.split()))

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


def normalize_feature_vectors(X):
    nan_feature_mean = []

    for feature_idx in range(X.shape[1]):
        nan_checker = np.isnan(X[:, feature_idx])
        if 0 < nan_checker.sum():
            # replace nan to avg.
            feature_mean = np.mean(X[np.nonzero(~nan_checker), feature_idx])
            X[np.nonzero(nan_checker), feature_idx] = feature_mean
            nan_feature_mean.append((feature_idx, feature_mean))

    return nan_feature_mean

nan_feature_mean = normalize_feature_vectors(X)
print(nan_feature_mean)

print('feature mean: ', np.mean(X, axis=0))
print('feature std: ', np.std(X, axis=0))

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=1000)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000,
                                    verbose=1)

# from sklearn.ensemble import GradientBoostingClassifier
# classifier = GradientBoostingClassifier(n_estimators=1000,
#                                         verbose=1)

classifier.fit(X, Y)

print(classifier.score(X, Y))
print(classifier.feature_importances_)

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
            sample_X.append(make_feature_vector(aid, pid, normalize_info=nan_feature_mean))
        sample_db[aid] = sample_X

    ok_idx = classifier.classes_.tolist().index(1)

    prob_Y = classifier.predict_proba(sample_X)

    return pids[np.argsort(prob_Y[:, ok_idx])[::-1]]


with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    f.write('AuthorId,PaperIds\n')
    for aid, pids in queries:
        print(aid)
        new_pids = sort_papers(aid, pids)
        f.write('{},{}\n'.format(
            aid,
            ' '.join(
                map(str, new_pids)
            )
        ))

with open(SAMPLE_DB_FILE, 'wb') as f:
    serializer.dump(sample_db, f, protocol=-1)
