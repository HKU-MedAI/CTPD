 CUDA_VISIBLE_DEVICES=3 python train_mimic3.py --task pheno --model_name CTPD --devices 1 \
    --use_multiscale --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 
