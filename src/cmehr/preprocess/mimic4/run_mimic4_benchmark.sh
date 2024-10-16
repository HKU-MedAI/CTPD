# python -m mimic4benchmark.scripts.create_multimodal_mimic4 --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/cxr  \
#     --cxr_path /disk1/*/EHR_dataset/mimiciv_benchmark/cxr/admission_w_cxr.csv

# python -m mimic4benchmark.scripts.create_24h_phenotyping --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/pheno
# python -m mimic4benchmark.scripts.create_24h_oud --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/oud
# python -m mimic4benchmark.scripts.create_24h_delirium --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/delirium
# python -m mimic4benchmark.scripts.create_in_hospital_mortality --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/ihm
# python -m mimic4benchmark.scripts.create_readmission_30d --root_path /disk1/*/EHR_dataset/mimiciv_benchmark \
#     --output_path /disk1/*/EHR_dataset/mimiciv_benchmark/readm
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/cxr
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/ihm
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/pheno
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/oud
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/delirium
# python -m mimic4models.split_train_val --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark/readm
# python -m mimic4models.create_irregular_ts --task pheno --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark
# python -m mimic4models.create_irregular_ts --task delirium --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark
# python -m mimic4models.create_irregular_ts --task oud --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark
# python -m mimic4models.create_irregular_ts --task ihm --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark
# python -m mimic4models.create_irregular_ts --task readm --dataset_dir /disk1/*/EHR_dataset/mimiciv_benchmark