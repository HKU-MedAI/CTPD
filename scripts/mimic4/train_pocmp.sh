CUDA_VISIBLE_DEVICES=6,7 python train_mimic4.py --task ihm --model_name CTPD --devices 2 --batch_size 16  \
    --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 
CUDA_VISIBLE_DEVICES=6,7 python train_mimic4.py --task pheno --model_name CTPD --devices 2 --batch_size 16 \
    --use_multiscale --use_prototype --lamb1 0.1 --lamb2 0.5 --lamb3 0.5 