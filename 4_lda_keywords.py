import datetime
import logging
import numpy as np
import os
import pickle as serializer
import sys

from gensim import corpora
from gensim.models.ldamodel import LdaModel

PUBLICATION_KEYWORDS_FILE = './publication_keywords.dump'
DICTIONARY_FILE = './keywords.dict'
CORPUS_FILE = './corpus.mm'
LDA_FILE = './result.lda'
TOPIC_FILE = './lda_topic.dump'

NUM_TOPICS = 500


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

corpus = corpora.MmCorpus(CORPUS_FILE)


# Make / Load LDA result

def make_lda_result():
    lda = LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=2, iterations=1000)

    # save LDA result
    lda.save(LDA_FILE)


if os.path.isfile(LDA_FILE):
    if input('Do you want to reload LDA result? (yes|otherwise)') == 'yes':
        make_lda_result()
else:
    make_lda_result()

lda = LdaModel.load(LDA_FILE)


# 4. Make and Save topic belief for each publication


with open(PUBLICATION_KEYWORDS_FILE, 'rb') as f:
    publication_keywords = serializer.load(f)

topic_result = dict()

for i, (pub_id, keywords) in enumerate(publication_keywords.items()):
    pub_topic = dict(lda[corpus[i]])
    if len(pub_topic) == 0:
        continue
    topic_belief = np.array([pub_topic.get(j, 0.0) for j in range(NUM_TOPICS)])
    topic_result[pub_id] = topic_belief

with open(TOPIC_FILE, 'wb') as f:
    serializer.dump((NUM_TOPICS, topic_result), f)
