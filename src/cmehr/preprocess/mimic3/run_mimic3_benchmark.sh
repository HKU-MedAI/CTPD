# python -m mimic3benchmark.scripts.create_multimodal_mimic3 --root_path /data1/r20user2/EHR_dataset/mimiciii_benchmark \
#     --output_path /data1/r20user2/EHR_dataset/mimiciii_benchmark/multimodal

# python -m mimic3models.split_train_val --dataset_dir /data1/r20user2/EHR_dataset/output_mimic3/self_supervised_multimodal

python -m mimic3models.create_irregular_multimodal --dataset_path /data1/r20user2/EHR_dataset/mimiciii_benchmark/multimodal