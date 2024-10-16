'''
This script is used to create self-supervised multimodal pairs from MIMIC-III dataset.
'''
import argparse
import os
import pandas as pd
import random
from tqdm import tqdm
import ipdb
from cmehr.preprocess.mimic3.mimic3models.text_utils import TextReader
from cmehr.paths import *
random.seed(49297)

'''
python -m mimic4benchmark.scripts.create_multimodal_mimic3 --root_path /disk1/*/EHR_dataset/mimiciii_benchmark \
    --output_path /disk1/*/EHR_dataset/mimiciii_benchmark/multimodal
'''

parser = argparse.ArgumentParser(description='Create multimodal pairs from MIMIC-III dataset.')
parser.add_argument('--root_path', type=str, default='data/mimic-iii-1.0', help='Path to MIMIC-III dataset.')
parser.add_argument('--output_path', type=str, default='data/mimic-iii-1.0-multimodal', help='Path to store the created multimodal pairs.')
args = parser.parse_args()


train_textdata_fixed = MIMIC3_BENCHMARK_PATH / "train_text_fixed"
train_starttime_path = MIMIC3_BENCHMARK_PATH / "train_starttime.pkl"
test_textdata_fixed = MIMIC3_BENCHMARK_PATH / "test_text_fixed"
test_starttime_path = MIMIC3_BENCHMARK_PATH / "test_starttime.pkl"


def process_partition(args, partition, eps=1e-6, n_hours=24):
    output_dir = os.path.join(args.output_path, partition)
    os.makedirs(output_dir, exist_ok=True)
    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(
        os.path.join(args.root_path, partition))))
    if partition == "train":
        text_reader = TextReader(train_textdata_fixed, train_starttime_path)
    else:
        text_reader = TextReader(test_textdata_fixed, test_starttime_path)

    # retrieve all available names
    available_names = []
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find(
            "timeseries") != -1, os.listdir(patient_folder)))
        patient_names = [patient + "_" + ts_filename for ts_filename in patient_ts_files]
        available_names.extend(patient_names)

    # read all text data within 48 hours to make it ...
    data_text, data_times, data_time = text_reader.read_all_text_append_json(
        available_names, 1e+6)
    
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find(
            "timeseries") != -1, os.listdir(patient_folder)))
        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(
                    os.path.join(patient_folder, lb_filename))
                # empty label file
                if label_df.shape[0] == 0:
                    continue

                icustay = label_df['Icustay'].iloc[0]

                # if no text data, skip
                if patient + "_" + ts_filename not in data_text:
                    continue
                text = data_text[patient + "_" + ts_filename]
                if len(text) == 0:
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

                xy_pairs.append((output_ts_filename, icustay))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id\n')
        for (x, icustay) in xy_pairs:
            listfile.write('{},0,{}\n'.format(x, icustay))


def main():
    os.makedirs(args.output_path, exist_ok=True)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()