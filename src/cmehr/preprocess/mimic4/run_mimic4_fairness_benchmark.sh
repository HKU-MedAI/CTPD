# python -m mimic4benchmark.scripts.extract_subjects --mimic4_path /disk1/fywang/EHR_dataset/mimiciv/tables \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4benchmark.scripts.validate_events --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4benchmark.scripts.extract_episodes_from_subjects --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --d_items_path /disk1/fywang/EHR_dataset/mimiciv/tables/d_items.csv 
# python -m mimic4benchmark.scripts.split_train_and_test --subjects_root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4benchmark.scripts.create_24h_phenotyping --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/pheno
# python -m mimic4benchmark.scripts.create_24h_oud --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/sud
# python -m mimic4benchmark.scripts.create_24h_delirium --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/delirium
# python -m mimic4benchmark.scripts.create_in_hospital_mortality --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/ihm
# python -m mimic4benchmark.scripts.create_readmission_30d --root_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
#     --output_path /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/readm
# python -m mimic4models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/ihm
# python -m mimic4models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/pheno
# python -m mimic4models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/sud
# python -m mimic4models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/delirium
# python -m mimic4models.split_train_val --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark/readm
# python -m mimic4models.create_irregular_ts --task pheno --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4models.create_irregular_ts --task delirium --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4models.create_irregular_ts --task sud --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4models.create_irregular_ts --task ihm --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark
# python -m mimic4models.create_irregular_ts --task readm --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark


python -m mimic4models.create_irregular_ts --task pheno  --modality_type TS CXR \
    --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
    --cxr_csv_path /home/fywang/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv

python -m mimic4models.create_irregular_ts --task delirium  --modality_type TS CXR \
    --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
    --cxr_csv_path /home/fywang/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv

python -m mimic4models.create_irregular_ts --task sud  --modality_type TS CXR \
    --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
    --cxr_csv_path /home/fywang/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv

python -m mimic4models.create_irregular_ts --task ihm  --modality_type TS CXR \
    --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
    --cxr_csv_path /home/fywang/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv

python -m mimic4models.create_irregular_ts --task readm  --modality_type TS CXR \
    --dataset_dir /disk1/fywang/EHR_dataset/mimiciv_fairness_benchmark \
    --cxr_csv_path /home/fywang/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv