import numpy as np
import os
import pickle as serializer

from config import *
from models import *


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

        # aid to pids (in paper_authors)
        np.matrix.sort(paper_authors, order=['aid', 'pid', 'nid'])

        aid_to_pids = dict()
        old_aid = -1
        for pid, aid, nid in paper_authors:
            if old_aid != aid:
                old_aid = aid
                aid_to_pids[aid] = []
            aid_to_pids[aid].append(pid)

        aid_to_nids = dict(((k, np.array(v)) for k, v in aid_to_nids.items()))
        nid_to_pids = dict(((k, np.array(v)) for k, v in nid_to_pids.items()))
        aid_to_pids = dict(((k, np.array(v)) for k, v in aid_to_pids.items()))

        with open(INDEX_FILE, 'wb') as f:
            serializer.dump((aid_to_nids, nid_to_pids, aid_to_pids), f, protocol=-1)

    if not os.path.isfile(INDEX_FILE):
        build_joint_table()

    with open(INDEX_FILE, 'rb') as f:
        aid_to_nids, nid_to_pids, aid_to_pids = serializer.load(f)

    def calculator(aid, pid):
        nids_arr = aid_to_nids[aid]
        unique_nids = np.unique(nids_arr)

        pids = aid_to_pids[aid]
        target_pid_idx = list(pids).index(pid)

        pids_traspose = pids.reshape(len(pids), 1)
        prob = np.zeros(len(pids))

        for nid in unique_nids:
            pids_given_nid_arr = nid_to_pids[nid]
            pids_given_nid_tile = np.tile(pids_given_nid_arr, (len(pids), 1))
            pid_nid_cnt = np.sum(pids_given_nid_tile == pids_traspose, axis=1)
            p1 = pid_nid_cnt / len(pids_given_nid_arr)

            p2 = 1.0 * np.sum(nids_arr == nid) / len(nids_arr)

            prob += p1 * p2

        return prob[target_pid_idx] / prob.sum()

    return calculator


def AuthorNameDiffer():
    from jellyfish import jaro_winkler

    def calculator(aid, pid):
        a_row = authors.get(aid)
        pa_row = paper_authors.get(pid, aid)

        if a_row is None or pa_row is None:
            return np.nan

        if (a_row[Authors.IDX_NAME] == '' or
                pa_row[PaperAuthors.IDX_NAME]) == '':
            return np.nan

        # aloready normalized name
        sim = jaro_winkler(
            a_row[Authors.IDX_NAME],
            pa_row[PaperAuthors.IDX_NAME]
        )
        return sim

    return calculator


def AuthorLastnameDiffer():
    from jellyfish import jaro_winkler

    def calculator(aid, pid):
        a_row = authors.get(aid)
        pa_row = paper_authors.get(pid, aid)

        if a_row is None or pa_row is None:
            return np.nan

        if (a_row[Authors.IDX_NAME] == '' or
                pa_row[PaperAuthors.IDX_NAME]) == '':
            return np.nan

        # aloready normalized name
        sim = jaro_winkler(
            a_row[Authors.IDX_NAME].split()[-1],
            pa_row[PaperAuthors.IDX_NAME].split()[-1]
        )
        return sim

    return calculator


def AuthorCoauthorNameDiffer():
    from jellyfish import jaro_winkler

    def calculator(aid, pid):
        a_row = authors.get(aid)

        if a_row is None or a_row[Authors.IDX_NAME] == '':
            return np.nan

        coa_dists = []
        for ipid, iaid in paper_authors.get_by_pid(pid):
            coa_row = authors.get(iaid)
            if coa_row is None or coa_row[Authors.IDX_NAME] == '':
                continue
            sim = jaro_winkler(
                a_row[Authors.IDX_NAME],
                coa_row[Authors.IDX_NAME]
            )
            coa_dists.append(sim)

        if not coa_dists:
            return np.nan

        return np.max(coa_dists)

    return calculator


def AuthorNameAbbChecker():
    import re

    def calculator(aid, pid):
        a_row = authors.get(aid)
        pa_row = paper_authors.get(pid, aid)

        if a_row is None or pa_row is None:
            return np.nan

        a_name = a_row[Authors.IDX_NAME].replace(' ', '')
        p_name = pa_row[PaperAuthors.IDX_NAME].replace(' ', '')

        if a_name == '' or p_name == '':
            return np.nan

        if a_name == p_name:
            return 1.0

        if (re.match(a_name.replace('.', '\w*'), p_name) or
                re.match(p_name.replace('.', '\w*'), a_name)):
            return 1.0

        return 0.0

    return calculator


def AffiliationNameDiffer():
    from jellyfish import jaro_winkler
    from unidecode import unidecode

    def calculator(aid, pid):
        a_row = authors.get(aid)
        pa_row = paper_authors.get(pid, aid)

        if a_row is None or pa_row is None:
            return np.nan

        if (a_row[Authors.IDX_AFF] == '' or
                pa_row[PaperAuthors.IDX_AFF]) == '':
            return np.nan

        sim = jaro_winkler(
            unidecode(a_row[Authors.IDX_AFF]).lower(),
            unidecode(pa_row[PaperAuthors.IDX_AFF]).lower()
        )
        return sim

    return calculator


def AuthorYearMid():
    def calculator(aid, pid):
        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            return np.median(years)

        return np.nan

    return calculator


def AuthorYearMin():
    def calculator(aid, pid):
        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            return np.min(years)

        return np.nan

    return calculator


def AuthorYearMax():
    def calculator(aid, pid):
        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            return np.max(years)

        return np.nan

    return calculator


def PaperYear():
    def calculator(aid, pid):
        p_row = papers.get(pid)

        if p_row is None:
            return np.nan

        if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):  # filter invalid info
            return np.nan

        return p_row[Papers.IDX_YEAR]

    return calculator


def AuthorNumPaper():
    def calculator(aid, pid):
        pid_aid_list = paper_authors.get_by_aid(aid)
        return len(pid_aid_list)

    return calculator


def PaperNumAuthor():
    def calculator(aid, pid):
        pid_aid_list = paper_authors.get_by_pid(pid)
        return len(pid_aid_list)

    return calculator


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


def AuthorNumPublication():
    def calculator(aid, pid):
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


def AuthorPaperTopicSim():
    import nltk

    from gensim import corpora
    from gensim import matutils
    from gensim.models.ldamodel import LdaModel
    from nltk.corpus import stopwords
    from unidecode import unidecode

    TOPIC_FILE = './lda_topic.dump'
    LDA_FILE = './result.lda'
    DICTIONARY_FILE = './keywords.dict'

    with open(TOPIC_FILE, 'rb') as f:
        num_topics, topic_result = serializer.load(f)

    print('# of topics: ' + str(num_topics))

    lda = LdaModel.load(LDA_FILE)

    dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def _convert_topic_array_to_indexed(topic_array):
        nn = np.nonzero(topic_array)[0]
        return list(zip((nn + 1), topic_array[nn]))

    def calculator(aid, pid):
        paper = papers.get(pid)
        if paper is None or paper[Papers.IDX_PUB_ID] is None:
            return np.nan

        publication = publications.get(paper[Papers.IDX_PUB_ID])

        publication_topic = topic_result[publication[Publications.IDX_ORIGINAL_ID]]

        my_keywords = []

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_TITLE]).lower())
            if not keywords:
                continue
            my_keywords.extend(keywords)

        my_keywords = list(filter(lambda s: s not in stopwords_set, my_keywords))
        if not my_keywords:
            return np.nan

        my_topic = lda[dictionary.doc2bow(my_keywords)]

        '''
        # Use cosine distance
        publication_topic = _convert_topic_array_to_indexed(publication_topic)
        return matutils.cossim(my_topic, publication_topic)
        '''

        # Use Hellinger distance
        my_topic_array = matutils.sparse2full(my_topic, num_topics)
        sim = np.sqrt(0.5 * ((np.sqrt(my_topic_array) - np.sqrt(publication_topic)) ** 2).sum())

        return sim

    return calculator


def AuthorNumTitleWords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def calculator(aid, pid):
        my_keywords = []

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_TITLE]).lower())
            if not keywords:
                continue
            my_keywords.extend(keywords)

        return len(my_keywords)

    return calculator


def PaperNumTitleWords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def calculator(aid, pid):
        paper = papers.get(pid)
        keywords = set(tokenizer.tokenize(unidecode(paper[Papers.IDX_TITLE]).lower()))
        return len(keywords)

    return calculator


def AuthorNumKeywords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def calculator(aid, pid):
        my_keywords = []

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_KEYWORDS]).lower())
            if not keywords:
                continue
            my_keywords.extend(keywords)

        return len(my_keywords)

    return calculator


def PaperNumKeywords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def calculator(aid, pid):
        paper = papers.get(pid)
        keywords = set(tokenizer.tokenize(unidecode(paper[Papers.IDX_KEYWORDS]).lower()))
        return len(keywords)

    return calculator

extractor_names = []
extractors = []

extractor_names.append('BayesAuthorToPaper')
extractor_names.append('AuthorNameDiffer')
extractor_names.append('AuthorLastnameDiffer')
extractor_names.append('AuthorCoauthorNameDiffer')
extractor_names.append('AffiliationNameDiffer')
extractor_names.append('AuthorYearMid')
extractor_names.append('AuthorYearMin')
extractor_names.append('AuthorYearMax')
extractor_names.append('PaperYear')
extractor_names.append('AuthorNumPaper')
extractor_names.append('PaperNumAuthor')
extractor_names.append('AuthorNumCoauthor')
extractor_names.append('AuthorNumPublication')
# extractor_names.append('AuthorPaperTopicSim')
extractor_names.append('AuthorNumTitleWords')
extractor_names.append('PaperNumTitleWords')
extractor_names.append('AuthorNumKeywords')
extractor_names.append('PaperNumKeywords')

if (os.path.isfile(FEATURE_DB_FILE)
        and os.path.isfile(SAMPLE_DB_FILE)):
    print('No need to load feature extractor')
else:
    # init feature extractor
    for extractor_name in extractor_names:
        extractor_maker = locals()[extractor_name]
        print('init {}...'.format(extractor_name))
        extractors.append(extractor_maker())


def make_feature_vector(aid, pid, missing_value_info=None):
    feature_vec = np.array(
        [extractor(aid, pid) for extractor in extractors]
    )

    feature_vec = feature_vec.flatten()

    if missing_value_info is not None:
        for feature_idx, feature_mean in missing_value_info:
            if np.isnan(feature_vec[feature_idx]):
                feature_vec[feature_idx] = feature_mean

    return feature_vec
