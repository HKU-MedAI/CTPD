CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name cnn 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name lstm
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name utde 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name seft 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name mtand 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name ipnet 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name grud 
CUDA_VISIBLE_DEVICES=0 python train_mimic3.py --task readm --model_name dgm2 