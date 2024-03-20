import argparse
import yaml

from mimic4benchmark.mimic4csv import *
from mimic4benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic4benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(
    description='Extract per-subject data from MIMIC-IV CSV files.')
parser.add_argument('--mimic4_path', type=str, help='Directory containing MIMIC-IV CSV files.', required=False,
                    default='/home/fwu/Documents/Datasets/physionet.org/files/mimiciv/2.2/tables')
parser.add_argument('--output_path', type=str, help='Directory where per-subject data should be written.', required=False,
                    default='data/root')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['chartevents', 'labevents', 'outputevents'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(
                        __file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str,
                    help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose',
                    action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose',
                    action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true',
                    help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

# mimic-iv use lower case for table & column names
try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)
stays = read_icustays_table(args.mimic4_path)
if args.verbose:
    print('START:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.mimic4_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path,
                 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(
    args.output_path, 'diagnosis_counts.csv'))

phenotypes = add_hcup_ccs_2015_groups(
    diagnoses, yaml.safe_load(open(args.phenotype_definitions, 'r')))

make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)

if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['subject_id']],
                        left_on='subject_id', right_on='subject_id')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0],
          'stays and only', args.event_tables[0], 'table')

subjects = stays.subject_id.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['itemid'].unique()]) if args.itemids_file else None
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic4_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)
