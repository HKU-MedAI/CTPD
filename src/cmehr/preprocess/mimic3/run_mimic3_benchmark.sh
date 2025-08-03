# 1. Extract subjects from MIMIC-III CSVs
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
# 2. Extract events from MIMIC-III CSVs
python -m mimic3benchmark.scripts.validate_events data/root/
# 3. Extract episodes from subjects
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
# 4. Split episodes into train and test sets
python -m mimic3benchmark.scripts.split_train_and_test data/root/
# 5. Create in-hospital mortality and phenotyping datasets
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
# 6. Split the dataset into training and validation sets
python -m mimic3models.split_train_val {dataset-directory}
# 7. Create a benchmark dataset
python -m mimic3models.create_iiregular_ts --task {TASK}