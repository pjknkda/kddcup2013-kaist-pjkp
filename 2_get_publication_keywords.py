import csv
import collections
import nltk
import pickle as serializer
import re
from unidecode import unidecode

from nltk.corpus import stopwords

CONFERENCE_FILE = '../data/Conference.csv'
JOURNAL_FILE = '../data/Journal.csv'
PAPER_FILE = '../data/Paper.csv'

PUB_KEYWORDS_FILE = './publication_keywords.dump'

TARGET_WORDS = 'Title'
#TARGET_WORDS = 'Keyword'


def readCSV(filepath):
    fieldnames = []

    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        fieldnames = next(reader)

    def row_generator():
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            int_field_idxes = []

            for i in range(len(fieldnames)):
                if fieldnames[i].endswith('Id'):
                    int_field_idxes.append(i)

            for row in reader:
                for int_field_idx in int_field_idxes:
                    row[int_field_idx] = int(row[int_field_idx])

                yield (row[0], row)

        print('read complete : {}'.format(filepath))

    return fieldnames, row_generator()

# load CSV files

conference_fields, conferences = readCSV(CONFERENCE_FILE)
journal_fields, journals = readCSV(JOURNAL_FILE)
paper_fields, papers = readCSV(PAPER_FILE)

publication_keywords = collections.defaultdict(list)


# tokenize and collect keywords

tokenizer = nltk.tokenize.RegexpTokenizer(r'[\w]{2,}')
stopwords_set = set(stopwords.words())

paper_keyword_idx = paper_fields.index(TARGET_WORDS)
paper_conference_id_idx = paper_fields.index('ConferenceId')
paper_journal_id_idx = paper_fields.index('JournalId')

for paper_id, paper in papers:
    keywords = tokenizer.tokenize(unidecode(paper[paper_keyword_idx]).lower())

    if not keywords:
        continue

    keywords = list(set(keywords) - stopwords_set)

    try:
        conference_id = paper[paper_conference_id_idx]
        if 0 < conference_id:
            publication_keywords[conference_id].extend(keywords)
    except Exception as ex:
        print('[Exception] {}'.format(paper))

    try:
        journal_id = paper[paper_journal_id_idx]
        if 0 < journal_id:
            publication_keywords[-journal_id].extend(keywords)
    except Exception as ex:
        print('[Exception] {}'.format(paper))


with open(PUB_KEYWORDS_FILE, 'wb') as f:
    serializer.dump(publication_keywords, f, protocol=-1)
