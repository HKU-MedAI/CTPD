'''
This script is used to create self-supervised multimodal pairs from MIMIC-IV dataset.
'''
import argparse
import os
import pandas as pd
import random
from tqdm import tqdm
import ipdb
random.seed(49297)

'''
python -m mimic4benchmark.scripts.create_multimodal_mimic4 --root_path /disk1/*/EHR_dataset/mimiciv_fairness_benchmark \
    --output_path /disk1/*/EHR_dataset/mimiciv_fairness_benchmark/cxr  \
    --cxr_path /disk1/*/EHR_dataset/mimiciv_benchmark/cxr/admission_w_cxr.csv
'''

parser = argparse.ArgumentParser(description='Create multimodal pairs from MIMIC-IV dataset.')
parser.add_argument('--root_path', type=str, default='data/mimic-iv-1.0', help='Path to MIMIC-IV dataset.')
parser.add_argument('--output_path', type=str, default='data/mimic-iv-1.0-multimodal', help='Path to store the created multimodal pairs.')
parser.add_argument('--cxr_path', type=str, default='data/mimic-iv-1.0-cxr', help='Path of CXR metadata.')
args = parser.parse_args()


def process_partition(args, partition, eps=1e-6, n_hours=24):
    output_dir = os.path.join(args.output_path, partition)
    os.makedirs(output_dir, exist_ok=True)
    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(
        os.path.join(args.root_path, partition))))
    cxr_df = pd.read_csv(args.cxr_path)
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find(
            "timeseries") != -1, os.listdir(patient_folder)))
        sub_cxr_df = cxr_df[cxr_df['subject_id'] == int(patient)]

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(
                    os.path.join(patient_folder, lb_filename))
                # empty label file
                if label_df.shape[0] == 0:
                    continue

                icustay = label_df['Icustay'].iloc[0]
                stay_cxr_df = sub_cxr_df[sub_cxr_df['stay_id'] == icustay]
                stay_cxr_df = stay_cxr_df.drop_duplicates(subset=['study_id'])

                # empty cxr file
                if stay_cxr_df.shape[0] == 0:
                    continue

                # mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)",
                          patient, ts_filename)
                    continue
                
                # make sure we have at least one day data
                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # keep events within LOS
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                imgids = stay_cxr_df['dicom_id'].tolist()
                xy_pairs.append((output_ts_filename, icustay, imgids))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,dicom_ids\n')
        for (x, icustay, dicom_ids) in xy_pairs:
            listfile.write('{},0,{},{}\n'.format(x, icustay, dicom_ids))


def main():
    os.makedirs(args.output_path, exist_ok=True)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()