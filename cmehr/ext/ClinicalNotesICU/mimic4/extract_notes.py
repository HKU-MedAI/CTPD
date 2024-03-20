from nltk import sent_tokenize, word_tokenize
import json
import re
import argparse
import os
import pandas as pd
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--note_csv', type=str,
                    default='/disk1/**/EHR_dataset/mimiciv/note/discharge.csv.gz')
args = parser.parse_args()

"""
Preprocess PubMed abstracts or MIMIC-III reports
"""

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()


def filter_for_first_hrs(dataframe, _days=2):
    min_time = dataframe.CHARTTIME.min()
    return dataframe[dataframe.CHARTTIME < min_time + pd.Timedelta(days=_days)]


def getText(t):
    return " ".join(list(preprocess_mimic(t)))


def getSentences(t):
    return list(preprocess_mimic(t))


df = pd.read_csv(args.note_csv)
df.charttime = pd.to_datetime(df.charttime)
df.storetime = pd.to_datetime(df.storetime)

df2 = df[df.subject_id.notnull()]
df2 = df2[df2.hadm_id.notnull()]
df2 = df2[df2.charttime.notnull()]
df2 = df2[df2.text.notnull()]
df2 = df2[['subject_id', 'hadm_id', 'charttime', 'text']]
del df
print(df2.groupby('hadm_id').count().describe())

'''
       subject_id  charttime      text
count    331794.0   331794.0  331794.0
mean          1.0        1.0       1.0
std           0.0        0.0       0.0
min           1.0        1.0       1.0
25%           1.0        1.0       1.0
50%           1.0        1.0       1.0
75%           1.0        1.0       1.0
max           1.0        1.0       1.0
'''

admission_df = pd.read_csv(
    "/disk1/**/EHR_dataset/mimiciv/hosp/admissions.csv.gz"
)
subject_ids = list(admission_df.subject_id.unique())

suceed = 0
failed = 0
patient_id = subject_ids[100]
# for patient_id in subject_ids:
#     sliced = df2[df2.subject_id == patient_id]
#     if sliced.shape[0] == 0:
#         print("No notes for PATIENT_ID : {}".format(patient_id))
#         failed += 1
#         continue
#     sliced.sort_values(by='charttime', inplace=True)

#     admission_df[admission_df.subject_id == patient_id]
# It seems that one admission only have one clinical note in mimic4, which is different from mimic3.
ipdb.set_trace()
