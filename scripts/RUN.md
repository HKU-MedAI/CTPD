## Instructions to run MMMSPG

1. Pretrain a self-supervised model to learn representations.

```bash
CUDA_VISIBLE_DEVICES=0,3 python pretrain_mimic4.py --devices 2
```

2. Extract embeddings from the training set.

```bash
CUDA_VISIBLE_DEVICES=0 python extract_pretrained_embs.py \
    --ckpt_path /home/fywang/Documents/EHR_codebase/MMMSPG/log/ckpts/mimic4_pretrain_2024-06-18_23-26-33/epoch=84-step=17510.ckpt
```
Here `ckpt_path` is the path of pretrained checkpoint in step 1.

BTW, this script can also be used to evaluate the performance of self-supervised representatioins. 

3. Cluster learned representations into clusters. 

```bash
python create_prototypes.py --n_proto 32
```

4. Learn prototype-aggregated representations.

It seems that svm achieves the best evaluation performance.
```bash
CUDA_VISIBLE_DEVICES=0 python embedding_mimic4.py --eval_method svm \
--ts_proto_path /home/fywang/Documents/EHR_codebase/MMMSPG/prototype_results/mimic4_pretrain/ts_proto_32.pkl \
--cxr_proto_path /home/fywang/Documents/EHR_codebase/MMMSPG/prototype_results/mimic4_pretrain/ts_proto_32.pkl

CUDA_VISIBLE_DEVICES=0 python embedding_mimic4.py --eval_method linear \
--ts_proto_path /home/fywang/Documents/EHR_codebase/MMMSPG/prototype_results/mimic4_pretrain/mm_proto_32.pkl
```

5. Multimodal fusion 

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_multimodal.py
```

### Comparison with baselines
```
CUDA_VISIBLE_DEVICES=0 python train_mimic4.py --modeltype TS --task ihm --model_name cnn
```