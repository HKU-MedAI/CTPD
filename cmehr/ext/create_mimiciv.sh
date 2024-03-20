# python -m mimic4benchmark.scripts.extract_subjects --mimic4_path /disk1/**/EHR_dataset/mimiciv/tables --output_path /disk1/**/EHR_dataset/mimiciv_benchmark
# python -m mimic4benchmark.scripts.validate_events --subjects_root_path /disk1/**/EHR_dataset/mimiciv_benchmark
# python -m mimic4benchmark.scripts.extract_episodes_from_subjects --subjects_root_path /disk1/**/EHR_dataset/mimiciv_benchmark --d_items_path /home/**/Documents/CM-EHR/data/mimiciv/tables/d_items.csv 
# python -m mimic4benchmark.scripts.split_train_and_test /disk1/**/EHR_dataset/mimiciv_benchmark
python -m mimic4benchmark.scripts.create_in_hospital_mortality /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/in-hospital-mortality/
python -m mimic4benchmark.scripts.create_24h_phenotyping /disk1/**/EHR_dataset/mimiciv_benchmark /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping_24h
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/in-hospital-mortality/
python -m mimic4models.split_train_val /disk1/**/EHR_dataset/mimiciv_benchmark/phenotyping_24h
