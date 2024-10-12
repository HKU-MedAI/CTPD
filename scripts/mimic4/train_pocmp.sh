CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mimic4.py --task ihm --model_name pocmp --devices 4 --batch_size 12 --lamb2 0
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mimic4.py --task pheno --model_name pocmp --devices 4 --batch_size 12 --lamb2 0
