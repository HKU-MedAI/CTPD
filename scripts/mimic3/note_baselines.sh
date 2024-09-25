# CUDA_VISIBLE_DEVICES=3 python run_note_baselines.py --task ihm --model_name flat
# CUDA_VISIBLE_DEVICES=3 python run_note_baselines.py --task pheno --model_name flat
CUDA_VISIBLE_DEVICES=4 python run_note_baselines.py --task ihm --model_name hiertrans
CUDA_VISIBLE_DEVICES=4 python run_note_baselines.py --task pheno --model_name hiertrans