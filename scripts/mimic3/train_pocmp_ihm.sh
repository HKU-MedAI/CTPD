# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0 --lamb2 0 --lamb3 0
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0 --lamb2 0.1 --lamb3 0.1 
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0 --lamb3 0
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1 
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 

CUDA_VISIBLE_DEVICES=6 python train_mimic3.py --task ihm --model_name pocmp_ts --devices 1 \
     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task ihm --model_name pocmp_note --devices 1 \
     --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.1 --lamb3 0.1

# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 0.5 --lamb2 0.5 --lamb3 0.5 
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 0.5 --lamb3 0.5
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 1 --lamb3 1
# CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task ihm --model_name pocmp --devices 1 \
#     --use_multiscale --use_prototype --lamb1 1 --lamb2 2 --lamb3 2
