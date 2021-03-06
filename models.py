import collections
import csv
import os
import pickle as serializer

from config import *


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
            self.rows, self.pub_id_mapping = _data
            return

        def _convert_row_JK_MANSE(row):
            return tuple([
                row[1],
                int(row[2]),
                publications.inv_map(row[3], row[4]),
                row[5]
            ])

        self.rows = dict()
        self.pub_id_mapping = collections.defaultdict(list)

        for row in readCSV(PAPER_FILE):
            new_row = _convert_row_JK_MANSE(row)
            self.rows[row[0]] = new_row

            self.pub_id_mapping[new_row[self.IDX_PUB_ID]].append(row[0])

    @property
    def data(self):
        return (self.rows, self.pub_id_mapping)

    def get(self, pid):
        return self.rows.get(pid)

    def get_by_pub_id(self, pub_id):
        return self.pub_id_mapping[pub_id]


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
