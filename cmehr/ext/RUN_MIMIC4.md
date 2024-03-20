### Commands to run MIMIC IV benchmark

1. Extract subjects: 
```
python -m mimic4benchmark.scripts.extract_subjects_iv /disk1/**/EHR_dataset/mimiciv/tables /disk1/**/EHR_dataset/mimiciv_benchmark
```

Logs: 
```
START:
        stay_ids: 73181
        hadm_ids: 66239
        subject_ids: 50920
REMOVE ICU TRANSFERS:
        stay_ids: 67858
        hadm_ids: 61942
        subject_ids: 48250
REMOVE MULTIPLE STAYS PER ADMIT:
        stay_ids: 56908
        hadm_ids: 56908
        subject_ids: 45127
REMOVE PATIENTS AGE < 18:
        stay_ids: 56908
        hadm_ids: 56908
        subject_ids: 45127
```

2. 
```
python -m mimic4benchmark.scripts.validate_events /disk1/**/EHR_dataset/mimiciv_benchmark
```
Log:
```
Iterating over subjects: 100% | 45127/45127

n_events: 281737943
empty_hadm: 16674216
no_hadm_in_stay: 46429698
no_icustay: 18856226
recovered: 18856226
could_not_recover: 0
icustay_missing_in_stays: 6971467
```

3.
```
python -m mimic4benchmark.scripts.extract_episodes_from_subjects /disk1/**/EHR_dataset/mimiciv_benchmark
```

4. 
```
python -m mimic4benchmark.scripts.split_train_and_test /disk1/**/EHR_dataset/mimiciv_benchmark
```

5. 
```
python -m mimic4benchmark.scripts.create_in_hospital_mortality /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/in-hospital-mortality/
<!-- python -m mimic4benchmark.scripts.create_phenotyping /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping/ -->
python -m mimic4benchmark.scripts.create_24h_phenotyping /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping_24h

```

6. 
```
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/in-hospital-mortality/
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping_24h
```

#### Create metadata for CXR

```
python extract_cxr.py --task ihm
python extract_cxr.py --task pheno
```