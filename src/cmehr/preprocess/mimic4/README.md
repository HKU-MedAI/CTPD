
### Pipeline for MIMIC IV benchmark

We follow MIMIC III benchmark to create this benchmark for MIMIC IV.

1. Before starting, we uncompress all csv files in `hosp` and `icu`, and move them into one folder `tables`.

2. The following command takes MIMIC-IV CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv` diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

```
python -m mimic4benchmark.scripts.extract_subjects --mimic4_path /disk1/fywang/EHR_dataset/mimiciv/tables --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
```

The output will be:
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

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows.
```
python -m mimic4benchmark.scripts.validate_events --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
```

The output will be: 
```
n_events: 308609511                                                                                                                      
empty_hadm: 17833104                                                                                                                     
no_hadm_in_stay: 51574120                                                                                                                
no_icustay: 19964949                                                                                                                     
recovered: 19964949                                                                                                                      
could_not_recover: 0                                                                                                                     
icustay_missing_in_stays: 8109819
```

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in `{SUBJECT_ID}/episode{#}_timeseries.csv` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in `{SUBJECT_ID}/episode{#}.csv`. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Outlier detection is disabled in the current version.
```
python -m mimic4benchmark.scripts.extract_episodes_from_subjects --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --d_items_path /disk1/fywang/EHR_dataset/mimiciv/tables/d_items.csv 
```

5. 

```
python -m mimic4benchmark.scripts.split_train_and_test --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
```


6. 
```
python -m mimic4benchmark.scripts.create_decompensation --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/decompensation/

python -m mimic4benchmark.scripts.create_length_of_stay /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/length-of-stay/

python -m mimic4benchmark.scripts.create_phenotyping /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping/
python -m mimic4benchmark.scripts.create_multitask /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/multitask/

python -m mimic4benchmark.scripts.create_in_hospital_mortality --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/ihm

python -m mimic4benchmark.scripts.create_24h_phenotyping --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/pheno

python -m mimic4benchmark.scripts.create_24h_oud --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/oud

python -m mimic4benchmark.scripts.create_24h_delirium --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/delirium

python -m mimic4benchmark.scripts.create_readmission_30d --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/readm
```

7.
```
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/in-hospital-mortality/
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping_24h
```

8. Preprocess csvs into pickle
```
XXX
```