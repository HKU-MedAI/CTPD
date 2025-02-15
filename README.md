# CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis

### Installation
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

### Dataset Preprocessing

1. Downdoad the original [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) database into your disk. 

2. Run the preprocessing code in `src/cmehr/preprocess/mimic3/run_mimic3_benchmark.sh`

Detailed steps: 

a. Run the command `cd src/cmehr/preprocess`.

b. The following command takes MIMIC-III CSVs, generates one directory per SUBJECT_ID and writes ICU stay information to data/{SUBJECT_ID}/stays.csv, diagnoses to data/{SUBJECT_ID}/diagnoses.csv, and events to data/{SUBJECT_ID}/events.csv. This step might take around an hour.
Here `data/root/` denotes the folder to store the processed benchmark data.

```
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

c. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows.
```
python -m mimic3benchmark.scripts.validate_events data/root/
```

d. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in {SUBJECT_ID}/episode{#}_timeseries.csv (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in {SUBJECT_ID}/episode{#}.csv. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Outlier detection is disabled in the current version.

```
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

e. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

```
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

f. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

```
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
```

g. Use the following command to extract validation set from the training set. This step is required for running the baseline models. Likewise, the train/test split, the train/validation split is the same for all tasks.

```
python -m mimic3models.split_train_val {dataset-directory}
```

f. create pickle files to store multimodal data.
```
python -m mimic3models.create_iiregular_ts --task {TASK}
``` 

Example commands can be found in `src/cmehr/preprocess/mimic3/run_mimic3_benchmark.sh`.

At the end, you will have a folder `output_mimic3`, which contains the stored pickle files for each task.

We thank awesome open-source repositories [MIMIC-III Benchmark](https://github.com/YerevaNN/mimic3-benchmarks), [ClinicalNotesICU](https://github.com/kaggarwal/ClinicalNotesICU) and [MultimodalMIMIC](https://github.com/XZhang97666/MultimodalMIMIC) for their code.

### Usage

Please see the scripts in `scripts/mimic3`.

To run our approach:
```
cd scripts/mimimc3
sh train_ctpd_ihm.sh
sh train_ctpd_pheno.sh
```

To run baselines:
```
sh ts_baseines.sh
sh note_baselines.sh
```

Note that the code of baselines is from their official repository.