from tqdm import tqdm
import os
import argparse
import pandas as pd
import yaml
import random
random.seed(49297)


def process_partition(args, delirium_codes,
                      partition, eps=1e-6, n_hours=24):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    total = 0
    pos = 0
    patients = list(filter(str.isdigit, os.listdir(
        os.path.join(args.root_path, partition))))
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

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)",
                          patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue
                # insurance = int(label_df.iloc[0]['Insurance'])
                # # in mimiciv, others is a category in insurance and race
                # if insurance == 0:
                #     continue
                # race = int(label_df.iloc[0]['Ethnicity'])
                # if race == 0:
                #     continue
                # gender = int(label_df.iloc[0]['Gender'])
                # if gender <= 0:
                #     continue
                # age = 1 if int(label_df.iloc[0]['Age']) >= 75 else 0

                # marital_status = int(label_df.iloc[0]['Marital_Status'])
                # if marital_status <= 0:
                #     continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                #             if -eps < t < los + eps]
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                delirium_label = 0
                total += 1

                icustay = label_df['Icustay'].iloc[0]
                diagnoses_df = pd.read_csv(os.path.join(patient_folder, "diagnoses.csv"),
                                           dtype={"icd_code": str})
                diagnoses_df = diagnoses_df[diagnoses_df.stay_id == icustay]
                for index, row in diagnoses_df.iterrows():
                    code = row['icd_code']
                    if code in delirium_codes:
                        delirium_label = 1
                        pos += 1
                xty_triples.append((output_ts_filename, icustay, n_hours, delirium_label))

    print("Number of created samples:", len(xty_triples))
    print("Number of positive samples:", pos)
    print("Number of total samples:", total)
    print("Percentage of positive samples: {:.2f}%".format(100.0 * pos / total))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "train":
        xty_triples = sorted(xty_triples)
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,stay_id,period_length,y_true\n')
        for (x, s, t, y) in xty_triples:
            listfile.write('{},{},{:.6f},{:d}\n'.format(x, s, t, y))


def main():
    parser = argparse.ArgumentParser(
        description="Create data for phenotype classification task.")
    parser.add_argument('--root_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/root/'),
                        help="Path to root folder containing train and test sets.")
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../../data/delirium/'),
                        help="Directory where the created data should be stored.")
    parser.add_argument('--phenotype_definitions', '-p', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                        help='YAML file with phenotype definitions.')
    args, _ = parser.parse_known_args()

    with open(args.phenotype_definitions) as definitions_file:
        definitions = yaml.safe_load(definitions_file)

    delirium_group = 'Delirium, dementia, and amnestic and other cognitive disorders'
    delirium_codes = definitions[delirium_group]['codes']

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, delirium_codes, "test")
    process_partition(args, delirium_codes, "train")


if __name__ == '__main__':
    main()