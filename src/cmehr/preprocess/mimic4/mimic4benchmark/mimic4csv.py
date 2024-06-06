import pdb
import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from cmehr.preprocess.mimic4.mimic4benchmark.util import dataframe_from_csv


nb_rows_dict = {'chartevents': 313645063,
                'labevents': 118171367,
                'outputevents': 4234967}
# nb_rows = {'chartevents': 330712484,
#            'labevents': 27854056,
#            'outputevents': 4349219}


def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'patients.csv'))
    pats = pats[['subject_id', 'gender', 'dod', 'anchor_age', 'anchor_year']]

    # mimic-iv deletes the DOB column in the patients table
    # pats.DOB = pd.to_datetime(pats.DOB)
    pats.dod = pd.to_datetime(pats.dod)
    return pats


def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(
        mimic4_path, 'admissions.csv'))
    # TODO: change this to keep more columns
    admits = admits[['subject_id', 'hadm_id', 'admittime',
                     'dischtime', 'deathtime', 
                     'admission_type', 'discharge_location',
                     'race', 'insurance', 'marital_status']]

    # mimic-iv deletes the DIAGNOSIS column in the admissions table
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits


def read_icustays_table(mimic4_path):
    stays = dataframe_from_csv(os.path.join(mimic4_path, 'icustays.csv'))
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays



def read_icd_diagnoses_table(path):

    codes = pd.read_csv(f'{path}/d_icd_diagnoses.csv')
    # dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['icd_code', 'long_title']]
    diagnoses = pd.read_csv(f'{path}/diagnoses_icd.csv')
    diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
    return diagnoses


def read_events_table_by_row(mimic4_path, table):
    reader = csv.DictReader(
        open(os.path.join(mimic4_path, table.lower() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        # mimic-iv use stay_id instead of icustay_id
        if 'stay_id' not in row:
            row['stay_id'] = ''
        yield row, i, nb_rows_dict[table.lower()]


def count_icd_codes(diagnoses, output_path=None):

    codes = diagnoses[['icd_code', 'long_title']].drop_duplicates().set_index('icd_code')
    codes['COUNT'] = diagnoses.groupby('icd_code')['stay_id'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    
    codes = codes[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label='icd_code')
    return codes.sort_values('COUNT', ascending=False).reset_index()


def remove_icustays_with_transfers(stays):
    # mimic-iv deletes FIRST_WARDID and LAST_WARDID columns
    stays = stays[stays.first_careunit == stays.last_careunit]
    # mimic-iv deletes DBSOURCE column
    return stays[['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'los']]


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])


def add_age_to_icustays(stays):
    # mimic-iv use anchor_age and anchor_year to calculate age
    stays['age'] = stays.apply(lambda e: (
        e['intime'].to_pydatetime()).year - e['anchor_year'] + e['anchor_age'], axis=1)
    stays.loc[stays.age >= 90, 'age'] = 90
    return stays


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.admittime <= stays.dod)
                                       & (stays.dischtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & (
        (stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)))
    stays['mortality'] = mortality.astype(int)
    stays['mortality_inhospital'] = stays['mortality']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.intime <= stays.dod)
                                       & (stays.outtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & (
        (stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime)))
    stays['mortality_inunit'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (
        to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    stays = stays.merge(to_keep, how='inner',
                        left_on='hadm_id', right_on='hadm_id')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.age >= min_age) & (stays.age <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def read_events_table_and_break_up_by_subject(mimic4_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['subject_id', 'hadm_id', 'stay_id',
                  'charttime', 'itemid', 'value', 'valueuom']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header,
                           quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    # This defines the number of rows in original table
    nb_rows = nb_rows_dict[table.lower()]

    for row, row_no, _ in tqdm(read_events_table_by_row(mimic4_path, table), total=nb_rows,
                               desc='Processing {} table'.format(table)):

        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['itemid'] not in items_to_keep):
            continue

        row_out = {'subject_id': row['subject_id'],
                   'hadm_id': row['hadm_id'],
                   'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                   'charttime': row['charttime'],
                   'itemid': row['itemid'],
                   'value': row['value'],
                   'valueuom': row['valueuom']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['subject_id']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']

    if data_stats.curr_subject_id != '':
        write_current_observations()
