import csv
import re
import itertools
from unidecode import unidecode

AUTHOR_FILE = '../data/Author.csv'
PAPER_AUTHOR_FILE = '../data/PaperAuthor.csv'

NORMALIZED_AUTHOR_FILE = './NormalizeAuthor.csv'
NORMALIZED_PAPER_AUTHOR_FILE = './NormalizePaperAuthor.csv'


def normalize(name):
    return unidecode(" ".join(name.strip().lower().split())).replace("-", " ")

assert normalize("j.-k. Park") == "j. k. park"
assert normalize("Kurt Gödel  ") == "kurt godel"
assert normalize("Kim Won jung") == "kim won jung"
assert normalize("J.   K.  Park") == "j. k. park"


with open(AUTHOR_FILE, 'r', encoding='utf-8', newline='') as input_file:
    with open(NORMALIZED_AUTHOR_FILE, 'w', encoding='utf-8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Column names
        column_names = next(input_file)
        output_file.write(column_names)

        for row in reader:
            row[1] = normalize(row[1])
            writer.writerow(row)

with open(PAPER_AUTHOR_FILE, 'r', encoding='utf-8') as input_file:
    with open(NORMALIZED_PAPER_AUTHOR_FILE, 'w', encoding='utf-8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Column names
        column_names = next(input_file)
        output_file.write(column_names)

        for row in reader:
            row[2] = normalize(row[2])
            writer.writerow(row)
