# CUDA_VISIBLE_DEVICES=2 python train_mimic3.py --task pheno --model_name pocmp --devices 1 --use_prototype --lamb1 0.
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 --use_multiscale
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1 
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.5 --lamb2 0.5 --lamb3 0.5 
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 0.5 --lamb3 0.5
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 1 --lamb3 1
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 2 --lamb3 2

# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0 --lamb2 0 --lamb3 0
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0 --lamb2 0.1 --lamb3 0.1 
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1 --num_slots 8
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 1 --lamb3 1
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 2 --lamb3 2
# CUDA_VISIBLE_DEVICES=1 python train_mimic3.py --task pheno --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1

CUDA_VISIBLE_DEVICES=7 python train_mimic3.py --task pheno --model_name pocmp_ts --devices 1 \
    --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1
# CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task pheno --model_name pocmp_note --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1