import numpy as np
import os
import pickle as serializer

from config import *
from models import *


def build_metadata_db():
    print('Building metadata DB')

    authors = Authors()

    publications = Publications()

    papers = Papers(publications=publications)

    paper_authors = PaperAuthors()

    with open(META_DB_FILE, 'wb') as f:
        serializer.dump((authors.data, publications.data, papers.data, paper_authors.data), f, protocol=-1)


if not os.path.isfile(META_DB_FILE):
    build_metadata_db()


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

    prob_cache_by_aid = [None, None]

    def calculator(aid, pid):
        nids_arr = aid_to_nids[aid]
        unique_nids = np.unique(nids_arr)

        pids = aid_to_pids[aid]
        target_pid_idx = list(pids).index(pid)

        if prob_cache_by_aid[0] == aid:
            prob = prob_cache_by_aid[1]
        else:
            pids_traspose = pids.reshape(len(pids), 1)
            prob = np.zeros(len(pids))

            for nid in unique_nids:
                pids_given_nid_arr = nid_to_pids[nid]
                pids_given_nid_tile = np.tile(pids_given_nid_arr, (len(pids), 1))
                pid_nid_cnt = np.sum(pids_given_nid_tile == pids_traspose, axis=1)
                p1 = pid_nid_cnt / len(pids_given_nid_arr)

                p2 = 1.0 * np.sum(nids_arr == nid) / len(nids_arr)

                prob += p1 * p2

            prob_cache_by_aid[0] = aid
            prob_cache_by_aid[1] = prob

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

        # already normalized name
        sim = jaro_winkler(
            a_row[Authors.IDX_NAME],
            pa_row[PaperAuthors.IDX_NAME]
        )
        return sim

    return calculator


def AuthorCoauthorNameDiffer():
    from jellyfish import jaro_winkler

    def calculator(aid, pid):
        a_row = authors.get(aid)

        if a_row is None or a_row[Authors.IDX_NAME] == '':
            return np.nan

        coa_sims = []
        for ipid, iaid in paper_authors.get_by_pid(pid):
            pa_row = paper_authors.get(pid, iaid)

            if pa_row is None or pa_row[PaperAuthors.IDX_NAME] == '':
                continue

            sim = jaro_winkler(
                a_row[Authors.IDX_NAME],
                pa_row[PaperAuthors.IDX_NAME]
            )
            coa_sims.append(sim)

        if not coa_sims:
            return np.nan

        return np.max(coa_sims)

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


def AuthorNameFormat():
    def calculator(aid, pid):
        a_row = authors.get(aid)

        if a_row is None:
            return np.nan

        a_name = a_row[Authors.IDX_NAME]

        if a_name == '':
            return np.nan

        return len(a_name.split())

    return calculator


def PaperNameFormat():
    def calculator(aid, pid):
        pa_row = paper_authors.get(pid, aid)

        if pa_row is None:
            return np.nan

        pa_name = pa_row[PaperAuthors.IDX_NAME]

        if pa_name == '':
            return np.nan

        return len(pa_name.split())

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


def AffiliationNameDiffer2():
    from jellyfish import levenshtein_distance
    from unidecode import unidecode

    def calculator(aid, pid):
        a_row = authors.get(aid)
        pa_row = paper_authors.get(pid, aid)

        if a_row is None or pa_row is None:
            return np.nan

        if (a_row[Authors.IDX_AFF] == '' or
                pa_row[PaperAuthors.IDX_AFF]) == '':
            return np.nan

        sim = levenshtein_distance(
            unidecode(a_row[Authors.IDX_AFF]).lower(),
            unidecode(pa_row[PaperAuthors.IDX_AFF]).lower()
        )
        return sim

    return calculator


def AuthorYearMid():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            result = np.median(years)
        else:
            result = np.nan

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

    return calculator


def AuthorYearMin():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            result = np.min(years)
        else:
            result = np.nan

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

    return calculator


def AuthorYearMax():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        years = []
        for ipid, iaid in paper_authors.get_by_aid(aid):
            p_row = papers.get(ipid)

            if p_row is None:
                continue
            if not (1800 <= p_row[Papers.IDX_YEAR] <= 2013):
                continue

            years.append(p_row[Papers.IDX_YEAR])

        if years:
            result = np.max(years)
        else:
            result = np.nan

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

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


def AuthorPaperDupNum():
    def calculator(aid, pid):
        aid_list = np.array([tu[1] for tu in paper_authors.get_by_pid(pid)])
        return np.sum(aid_list == aid)

    return calculator


def AuthorNumCoauthor():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            author_coauthors = cache_by_aid[1]
        else:
            author_coauthors = set()
            for ipid, iaid in paper_authors.get_by_aid(aid):
                for copid, coaid in paper_authors.get_by_pid(ipid):
                    author_coauthors.add(coaid)

            cache_by_aid[0] = aid
            cache_by_aid[1] = author_coauthors

        return len(author_coauthors)

    return calculator


def AuthorMidNumCoauthor():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            author_num_coauthors = cache_by_aid[1]
        else:
            author_num_coauthors = []
            for ipid, iaid in paper_authors.get_by_aid(aid):
                author_num_coauthors.append(len(paper_authors.get_by_pid(ipid)))

            cache_by_aid[0] = aid
            cache_by_aid[1] = author_num_coauthors

        return np.median(author_num_coauthors)

    return calculator


def AuthorMinNumCoauthor():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            author_num_coauthors = cache_by_aid[1]
        else:
            author_num_coauthors = []
            for ipid, iaid in paper_authors.get_by_aid(aid):
                author_num_coauthors.append(len(paper_authors.get_by_pid(ipid)))

            cache_by_aid[0] = aid
            cache_by_aid[1] = author_num_coauthors

        return np.min(author_num_coauthors)

    return calculator


def AuthorMaxNumCoauthor():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            author_num_coauthors = cache_by_aid[1]
        else:
            author_num_coauthors = []
            for ipid, iaid in paper_authors.get_by_aid(aid):
                author_num_coauthors.append(len(paper_authors.get_by_pid(ipid)))

            cache_by_aid[0] = aid
            cache_by_aid[1] = author_num_coauthors

        return np.max(author_num_coauthors)

    return calculator


def AuthorMeanNumCoauthor():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            author_num_coauthors = cache_by_aid[1]
        else:
            author_num_coauthors = []
            for ipid, iaid in paper_authors.get_by_aid(aid):
                author_num_coauthors.append(len(paper_authors.get_by_pid(ipid)))

            cache_by_aid[0] = aid
            cache_by_aid[1] = author_num_coauthors

        return np.mean(author_num_coauthors)

    return calculator


def AuthorPaperNumCoauthor():
    coauthor_cache_by_aid = [None, None]

    def calculator(aid, pid):
        target_paper_authors = set()
        for ipid, iaid in paper_authors.get_by_pid(pid):
            target_paper_authors.add(iaid)

        if coauthor_cache_by_aid[0] == aid:
            author_coauthors = coauthor_cache_by_aid[1]
        else:
            author_coauthors = set()
            for ipid, iaid in paper_authors.get_by_aid(aid):
                for copid, coaid in paper_authors.get_by_pid(ipid):
                    author_coauthors.add(coaid)

            coauthor_cache_by_aid[0] = aid
            coauthor_cache_by_aid[1] = author_coauthors

        return len(target_paper_authors & author_coauthors)

    return calculator


def AuthorNumPublication():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        my_publications = set()
        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            if paper[Papers.IDX_PUB_ID] is None:
                continue
            my_publications.add(paper[Papers.IDX_PUB_ID])

        result = len(my_publications)

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

    return calculator


def AuthorRatioPublicationJournal():
    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        total_publication = 0
        journal_publication = 0

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            if paper[Papers.IDX_PUB_ID] is None:
                continue

            total_publication += 1

            publication = publications.get(paper[Papers.IDX_PUB_ID])
            pub_ori_id = publication[Publications.IDX_ORIGINAL_ID]

            if pub_ori_id < 0:
                journal_publication += 1

        if total_publication == 0:
            result = 0.5
        else:
            result = 1.0 * journal_publication / total_publication

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

    return calculator


def AuthorNumTitleWords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        my_keywords = []

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_TITLE]).lower())
            if not keywords:
                continue
            my_keywords.extend(keywords)

        result = len(my_keywords)

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

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

    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        my_keywords = []

        for ipid, iaid in paper_authors.get_by_aid(aid):
            paper = papers.get(ipid)
            if paper is None:
                continue
            keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_KEYWORDS]).lower())
            if not keywords:
                continue
            my_keywords.extend(keywords)

        result = len(my_keywords)

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

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


def PaperAuthorIntersectKeywords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            my_keywords = cache_by_aid[1]

        else:
            my_keywords = []

            for ipid, iaid in paper_authors.get_by_aid(aid):
                paper = papers.get(ipid)
                if paper is None:
                    continue
                keywords = tokenizer.tokenize(unidecode(paper[Papers.IDX_KEYWORDS]).lower())
                if not keywords:
                    continue
                my_keywords.extend(keywords)

            my_keywords = set(my_keywords)

            cache_by_aid[0] = aid
            cache_by_aid[1] = my_keywords

        paper = papers.get(pid)
        keywords = set(tokenizer.tokenize(unidecode(paper[Papers.IDX_KEYWORDS]).lower()))

        return len(my_keywords & keywords)

    return calculator


def AuthorAffNumWords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    cache_by_aid = [None, None]

    def calculator(aid, pid):
        if cache_by_aid[0] == aid:
            return cache_by_aid[1]

        a_row = authors.get(aid)
        keywords = set(tokenizer.tokenize(unidecode(a_row[Authors.IDX_AFF]).lower()))

        result = len(keywords)

        cache_by_aid[0] = aid
        cache_by_aid[1] = result

        return result

    return calculator


def PaperAffNumWords():
    import nltk

    from nltk.corpus import stopwords
    from unidecode import unidecode

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    def calculator(aid, pid):
        pa_row = paper_authors.get(pid, aid)
        keywords = set(tokenizer.tokenize(unidecode(pa_row[PaperAuthors.IDX_AFF]).lower()))

        result = len(keywords)

        return result

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

    lda = LdaModel.load(LDA_FILE)

    dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    my_topic_cache_by_aid = [None, None]

    def calculator(aid, pid):
        paper = papers.get(pid)
        if paper is None or paper[Papers.IDX_PUB_ID] is None:
            return np.nan

        publication = publications.get(paper[Papers.IDX_PUB_ID])

        pub_ori_id = publication[Publications.IDX_ORIGINAL_ID]
        if pub_ori_id not in topic_result:
            return np.nan

        publication_topic = topic_result[pub_ori_id]

        if my_topic_cache_by_aid[0] == aid:
            my_topic = my_topic_cache_by_aid[1]
        else:
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

            my_topic_cache_by_aid[0] = aid
            my_topic_cache_by_aid[1] = my_topic

        # Use Hellinger distance
        my_topic_array = matutils.sparse2full(my_topic, num_topics)
        sim = np.sqrt(0.5 * ((np.sqrt(my_topic_array) - np.sqrt(publication_topic)) ** 2).sum())

        return sim

    return calculator


def AuthorTopicMean():
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

    lda = LdaModel.load(LDA_FILE)

    dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    my_topic_cache_by_aid = [None, None]

    def calculator(aid, pid):
        if my_topic_cache_by_aid[0] == aid:
            my_topic = my_topic_cache_by_aid[1]
        else:
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

            my_topic_cache_by_aid[0] = aid
            my_topic_cache_by_aid[1] = my_topic

        my_topic_array = matutils.sparse2full(my_topic, num_topics)
        return np.mean(my_topic_array)

    return calculator


def AuthorTopicStd():
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

    lda = LdaModel.load(LDA_FILE)

    dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
    stopwords_set = set(stopwords.words())

    my_topic_cache_by_aid = [None, None]

    def calculator(aid, pid):
        if my_topic_cache_by_aid[0] == aid:
            my_topic = my_topic_cache_by_aid[1]
        else:
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

            my_topic_cache_by_aid[0] = aid
            my_topic_cache_by_aid[1] = my_topic

        my_topic_array = matutils.sparse2full(my_topic, num_topics)
        return np.std(my_topic_array)

    return calculator


def PaperTopicMean():
    TOPIC_FILE = './lda_topic.dump'

    with open(TOPIC_FILE, 'rb') as f:
        num_topics, topic_result = serializer.load(f)

    def calculator(aid, pid):
        paper = papers.get(pid)
        if paper is None or paper[Papers.IDX_PUB_ID] is None:
            return np.nan

        publication = publications.get(paper[Papers.IDX_PUB_ID])

        pub_ori_id = publication[Publications.IDX_ORIGINAL_ID]
        if pub_ori_id not in topic_result:
            return np.nan

        publication_topic = topic_result[pub_ori_id]

        return np.mean(publication_topic)

    return calculator


def PaperTopicStd():
    TOPIC_FILE = './lda_topic.dump'

    with open(TOPIC_FILE, 'rb') as f:
        num_topics, topic_result = serializer.load(f)

    def calculator(aid, pid):
        paper = papers.get(pid)
        if paper is None or paper[Papers.IDX_PUB_ID] is None:
            return np.nan

        publication = publications.get(paper[Papers.IDX_PUB_ID])

        pub_ori_id = publication[Publications.IDX_ORIGINAL_ID]
        if pub_ori_id not in topic_result:
            return np.nan

        publication_topic = topic_result[pub_ori_id]

        return np.std(publication_topic)

    return calculator


def PublicationNumPaper():

    def calculator(aid, pid):
        paper = papers.get(pid)
        if paper is None or paper[Papers.IDX_PUB_ID] is None:
            return np.nan

        pub_id = paper[Papers.IDX_PUB_ID]

        return len(papers.get_by_pub_id(pub_id))

    return calculator


feature_names = []

# first feature
feature_names.append('BayesAuthorToPaper')
feature_names.append('AuthorNameDiffer')
feature_names.append('AffiliationNameDiffer')
feature_names.append('PaperYear')
feature_names.append('AuthorNumPaper')
feature_names.append('AuthorPaperNumCoauthor')

# second festure
feature_names.append('PaperNumAuthor')
feature_names.append('AuthorNumPublication')
feature_names.append('AuthorPaperTopicSim')

# third feature
# feature_names.append('AuthorNumCoauthor')
# feature_names.append('AuthorCoauthorNameDiffer'))
# feature_names.append('AuthorNumTitleWords')
# feature_names.append('PaperNumTitleWords')
# feature_names.append('AuthorNumKeywords')
# feature_names.append('PaperNumKeywords')
# feature_names.append('AuthorAffNumWords')
# feature_names.append('PaperAffNumWords')

# feature_names.append('AuthorYearMid')
# feature_names.append('AuthorYearMin')
# feature_names.append('AuthorYearMax')
# feature_names.append('AuthorMidNumCoauthor')
# feature_names.append('AuthorMinNumCoauthor')
# feature_names.append('AuthorMaxNumCoauthor')
# feature_names.append('AuthorMeanNumCoauthor')
# feature_names.append('AuthorRatioPublicationJournal')
# feature_names.append('AuthorTopicMean')
# feature_names.append('AuthorTopicStd')
# feature_names.append('PaperTopicMean')
# feature_names.append('PaperTopicStd')

# last feature
feature_names.append('PublicationNumPaper')
feature_names.append('AuthorPaperDupNum')
feature_names.append('PaperAuthorIntersectKeywords')
feature_names.append('AuthorNameFormat')
feature_names.append('PaperNameFormat')

is_all_cached = True

# check train features
for feature_name in feature_names:
    feature_dump_location = TRAIN_FEATURE_DB_FOLDER + '/{}.dump'.format(feature_name)
    if not os.path.isfile(feature_dump_location):
        is_all_cached = False
        break

# check test features
for filepath in [VALID_FILE, TEST_FILE]:
    target_feature_db_folder = TARGET_FEATURE_DB_FOLDER + filepath.split('/')[-1]
    for feature_name in feature_names:
        feature_dump_location = target_feature_db_folder + '/{}.dump'.format(feature_name)
        if not os.path.isfile(feature_dump_location):
            is_all_cached = False
            break

if is_all_cached:
    print('No need to load metadata')
else:
    with open(META_DB_FILE, 'rb') as f:
        authors_data, publications_data, papers_data, paper_authors_data = serializer.load(f)
        authors = Authors(_data=authors_data)
        publications = Publications(_data=publications_data)
        papers = Papers(_data=papers_data)
        paper_authors = PaperAuthors(_data=paper_authors_data)

    print('Loading metadata is completed.')


def get_feature_extractor(feature_name):
    extractor_maker = globals()[feature_name]
    extractor = extractor_maker()
    return extractor


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
