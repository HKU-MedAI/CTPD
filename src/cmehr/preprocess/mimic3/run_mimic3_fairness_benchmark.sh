# python -m mimic3benchmark.scripts.extract_subjects --mimic3_path /disk1/fywang/EHR_dataset/mimiciii \
#     --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
# python -m mimic3benchmark.scripts.validate_events --subjects_root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
# python -m mimic3benchmark.scripts.extract_episodes_from_subjects --subjects_root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3benchmark.scripts.create_in_hospital_mortality --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark \
    --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/ihm
python -m mimic3benchmark.scripts.create_24h_phenotyping --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark \
    --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/pheno
python -m mimic3benchmark.scripts.create_24h_delirium --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark \
    --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/delirium
python -m mimic3benchmark.scripts.create_24h_oud --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark \
    --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/sud
python -m mimic3benchmark.scripts.create_readmission_30d --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark \
    --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/readm
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/ihm
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/pheno
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/sud
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/delirium
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/readm
python -m mimic3models.create_iiregular_ts --task pheno --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task delirium --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task sud --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task ihm --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task readm --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark