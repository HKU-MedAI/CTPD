# CUDA_VISIBLE_DEVICES=4 python run_note_baselines.py --task ihm --model_name tlstm
# CUDA_VISIBLE_DEVICES=4 python run_note_baselines.py --task pheno --model_name tlstm
CUDA_VISIBLE_DEVICES=4 python run_note_baselines.py --task ihm --model_name ftlstm
CUDA_VISIBLE_DEVICES=3 python run_note_baselines.py --task pheno --model_name ftlstm