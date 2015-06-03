import datetime
import logging
import nltk
import os
import pickle as serializer

from gensim import corpora
from gensim.models.ldamodel import LdaModel

PUBLICATION_KEYWORDS_FILE = './publication_keywords.dump'
DICTIONARY_FILE = './keywords.dict'
CORPUS_FILE = './corpus.mm'

FILTER_NO_BELOW = 5
FILTER_NO_ABOVE = 0.25


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# Load publication info

with open(PUBLICATION_KEYWORDS_FILE, 'rb') as f:
    publication_keywords = serializer.load(f).items()


# 1. Make / Load dictionary

def make_dictionary():
    texts = []
    for pub_info, keywords in publication_keywords:
        if keywords:
            texts.append(keywords)

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=FILTER_NO_BELOW,
                               no_above=FILTER_NO_ABOVE)

    # save dictionary
    dictionary.save(DICTIONARY_FILE)


if os.path.isfile(DICTIONARY_FILE):
    if input('Do you want to reload dictionary? (yes|otherwise)') == 'yes':
        make_dictionary()
else:
    make_dictionary()

dictionary = corpora.Dictionary.load(DICTIONARY_FILE)

print(dictionary)


# 2. Make / Load corpus

def make_corpus():
    corpus = []
    for pub_info, keywords in publication_keywords:
        corpus.append(dictionary.doc2bow(keywords))

    # save corpuse
    corpora.MmCorpus.serialize(CORPUS_FILE, corpus)

if os.path.isfile(CORPUS_FILE):
    if input('Do you want to reload corpus? (yes|otherwise)') == 'yes':
        make_corpus()
else:
    make_corpus()

corpus = corpora.MmCorpus(CORPUS_FILE)

print(corpus)
