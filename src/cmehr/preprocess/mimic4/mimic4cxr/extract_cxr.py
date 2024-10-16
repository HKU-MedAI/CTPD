import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import ipdb

parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default='ihm',
#                     choices=["ihm", "pheno"])
parser.add_argument('--mimic_cxr_path', type=str,
                    default='/disk1/*/CXR_dataset/mimic_data/2.0.0')
parser.add_argument('--mimic_iv_admission_csv', type=str,
                    default="/disk1/*/EHR_dataset/mimiciv_benchmark/all_stays.csv")
parser.add_argument('--save_dir', type=str,
                    default="/disk1/*/EHR_dataset/mimiciv_benchmark/cxr")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)


def main():
    # part of code borrow from
    # https://github.com/nyuad-cai/MedFuse/blob/main/datasets/fusion.py
    icu_stay_metadata = pd.read_csv(args.mimic_iv_admission_csv)
    cxr_metadata = pd.read_csv(os.path.join(args.mimic_cxr_path,
                                            "mimic-cxr-2.0.0-metadata.csv"))
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']

    # only common subjects with both icu stay and an xray
    cxr_merged_icustays = cxr_metadata.merge(
        icu_stay_metadata[columns], how='inner', on='subject_id')

    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(
        lambda x: f'{int(float(x)):06}')
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(
        str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str), format="%Y%m%d %H%M%S")

    cxr_merged_icustays.intime = pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime = pd.to_datetime(cxr_merged_icustays.outtime)
    end_time = cxr_merged_icustays.outtime
    # if args.task == 'ihm':
    #     end_time = cxr_merged_icustays.intime + pd.Timedelta(hours=48)
    # elif args.task == 'pheno':
    #     end_time = cxr_merged_icustays.intime + pd.Timedelta(hours=24)

    # we keep cxrs within all icu stays
    cxr_merged_icustays_during = cxr_merged_icustays.loc[(
        cxr_merged_icustays.StudyDateTime >= cxr_merged_icustays.intime) & ((cxr_merged_icustays.StudyDateTime <= end_time))]

    # cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.outtime))]
    # select cxrs with the ViewPosition == 'AP' or 'PA'
    cxr_merged_icustays_AP = cxr_merged_icustays_during[
        cxr_merged_icustays_during['ViewPosition'].isin(['AP', 'PA'])]

    all_path = []
    for _, row in tqdm(cxr_merged_icustays_AP.iterrows(), total=len(cxr_merged_icustays_AP)):
        subject_id = row["subject_id"]
        study_id = row["study_id"]
        dicom_id = row["dicom_id"]
        path = os.path.join("p" + str(subject_id)[:2],
                            "p" + str(subject_id),
                            "s" + str(study_id),
                            dicom_id + ".jpg")
        full_path = os.path.join(args.mimic_cxr_path, "files", path)
        if not os.path.exists(full_path):
            continue
        all_path.append(path)

    cxr_merged_icustays_AP["path"] = all_path
    cxr_merged_icustays_AP.to_csv(
        os.path.join(args.save_dir, f"admission_w_cxr.csv"), index=False
    )


if __name__ == '__main__':
    main()