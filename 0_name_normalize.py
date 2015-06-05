import csv
import re
import itertools
from unidecode import unidecode
from nameparser import HumanName

AUTHOR_FILE = '../data/Author.csv'
PAPER_AUTHOR_FILE = '../data/PaperAuthor.csv'

NORMALIZED_AUTHOR_FILE = './NormalizeAuthor.csv'
NORMALIZED_PAPER_AUTHOR_FILE = './NormalizePaperAuthor.csv'


def normalize(name):
    norm_1 = unidecode(" ".join(name.strip().lower().split())).replace("-", " ")
    norm_2 = re.sub(r'<[^>]+>', r'', norm_1)  # remove html
    hname = HumanName(norm_2)
    hname.string_format = '{first} {middle} {last}'
    # return str(hname).replace()
    return re.sub(r'[^a-z\s]', r'', str(hname))

# assert normalize("j.-k. Park") == "j. k. park"
# assert normalize("Kurt Gödel  ") == "kurt godel"
# assert normalize("Kim Won jung") == "kim won jung"
# assert normalize("J.   K.  Park") == "j. k. park"


with open(AUTHOR_FILE, 'r', encoding='utf-8', newline='') as input_file:
    with open(NORMALIZED_AUTHOR_FILE, 'w', encoding='utf-8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Column names
        column_names = next(input_file)
        output_file.write(column_names)

        for i, row in enumerate(reader):
            if i % 1000 == 0:
                print(i)

            row[1] = normalize(row[1])
            writer.writerow(row)

with open(PAPER_AUTHOR_FILE, 'r', encoding='utf-8') as input_file:
    with open(NORMALIZED_PAPER_AUTHOR_FILE, 'w', encoding='utf-8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Column names
        column_names = next(input_file)
        output_file.write(column_names)

        for i, row in enumerate(reader):
            if i % 1000 == 0:
                print(i)

            row[2] = normalize(row[2])
            writer.writerow(row)
