# 1. The following command takes MIMIC-III CSVs, generates one directory per SUBJECT_ID and writes ICU stay information to data/{SUBJECT_ID}/stays.csv, diagnoses to data/{SUBJECT_ID}/diagnoses.csv, and events to data/{SUBJECT_ID}/events.csv. This step might take around an hour.
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/

python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_benchmark/readmission_30d
python -m mimic3models.create_iiregular_ts --task readm
