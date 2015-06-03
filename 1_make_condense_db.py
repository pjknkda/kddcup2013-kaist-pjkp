import csv
import numpy
import os
import pickle as serializer

AUTHOR_FILE = './NormalizeAuthor.csv'
PAPER_AUTHOR_FILE = './NormalizePaperAuthor.csv'
CONDENSE_DB_FILE = './condense_db.dump'


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
                if not row:
                    continue

                for int_field_idx in int_field_idxes:
                    row[int_field_idx] = int(row[int_field_idx])

                yield row

        print('read complete : {}'.format(filepath))

    return fieldnames, row_generator()


def make_condense_db():
    author_fields, authors = readCSV(AUTHOR_FILE)
    paper_author_fields, paper_authors = readCSV(PAPER_AUTHOR_FILE)

    name_id_mapping = dict()
    name_id_counter = 0

    condense_authors = []
    condense_paper_authors = []

    for author in authors:
        name = author[1]
        if name not in name_id_mapping:
            name_id_counter += 1
            name_id_mapping[name] = name_id_counter
        author[1] = name_id_mapping[name]

        condense_authors.append((
            author[0],
            author[1]
        ))

    for paper_author in paper_authors:
        name = paper_author[2]
        if name not in name_id_mapping:
            name_id_counter += 1
            name_id_mapping[name] = name_id_counter
        paper_author[2] = name_id_mapping[name]

        condense_paper_authors.append((
            paper_author[0],
            paper_author[1],
            paper_author[2]
        ))

    condense_authors_np = numpy.array(condense_authors, dtype=[('aid', int), ('nid', int)])
    condense_paper_authors_np = numpy.array(condense_paper_authors, dtype=[('pid', int), ('aid', int), ('nid', int)])

    condense_db = [
        condense_authors_np,
        condense_paper_authors_np,
        name_id_mapping
    ]

    with open(CONDENSE_DB_FILE, 'wb') as f:
        serializer.dump(condense_db, f, protocol=-1)


if os.path.isfile(CONDENSE_DB_FILE):
    if input('Do you want to rebuild DB? (yes|otherwise)') == 'yes':
        make_condense_db()
else:
    make_condense_db()
