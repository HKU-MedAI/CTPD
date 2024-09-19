# MSPG


### Installation
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Code for EHR pretraining.

- 1. Self-supervised learning from multi-modality data
- 2. Initialize prototypes
- 3. DiffEM for TS-level embedding extraction
- 4. Downstream task


### Usage 
```
git clone https://github.com/KaedeGo/MMMSPG
pip install -e .
```

### TODO

- [x] formalize dataset with different modalities
- [x] process notes
- [x] prototype learning
- [x] double check preprocessing code, which has been run in the cmehr repo.
- [x] collect a large dataset for self-supervised pretraining ...
- [x] multimodal fusion
- [x] merge dataset preprocessing code ... maybe write a README.md to illustrate how to formulate the dataset 
- [ ] explore frequency prototypes for time series.
- [ ] problem: the feature in temporal dimension is very similar.
- [ ] TODO: maybe construct a prototype map ...

### Results
XXX