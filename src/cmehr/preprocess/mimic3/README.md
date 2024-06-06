1. 

```
python -m mimic3benchmark.scripts.extract_subjects --mimic3_path /disk1/fywang/EHR_dataset/mimiciii --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
```

2.

```
python -m mimic3benchmark.scripts.validate_events --subjects_root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
```

3. 
```
python -m mimic3benchmark.scripts.extract_episodes_from_subjects --subjects_root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
```

4. 
```
python -m mimic3benchmark.scripts.split_train_and_test --subjects_root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
```

5. 
```
python -m mimic3benchmark.scripts.create_in_hospital_mortality --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_24h_phenotyping --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/phenotyping_24h
python -m mimic3benchmark.scripts.create_24h_delirium --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/delirium_24h
python -m mimic3benchmark.scripts.create_24h_oud --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/oud_24h
python -m mimic3benchmark.scripts.create_readmission_30d --root_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/readmission_30d
```

6. 
```
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/in-hospital-mortality/
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/phenotyping_24h
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/oud_24h
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/delirium_24h
python -m mimic3models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark/readmission_30d
```

7.
```
python -m mimic3models.create_iiregular_ts --task pheno --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task delirium --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task oud --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task ihm --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
python -m mimic3models.create_iiregular_ts --task readm --dataset_dir /disk1/fywang/EHR_dataset/mimiciii_fairness_benchmark
```