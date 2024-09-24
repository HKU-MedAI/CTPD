
import os
import argparse
from pathlib import Path


ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = Path("/disk1/fywang/EHR_dataset")
# DATA_PATH = Path("/data1/r10user2/EHR_dataset")
# DATA_PATH = Path("/data1/r20user2/EHR_dataset")

# path for mimic iii dataset
# only used for preprocessing benchmark dataset
MIMIC3_BENCHMARK_PATH = DATA_PATH / "mimiciii_benchmark"
MIMIC3_RAW_PATH = DATA_PATH / "mimiciii"
MIMIC3_IHM_PATH = MIMIC3_BENCHMARK_PATH / "in-hospital-mortality"
MIMIC3_PHENO_PATH = MIMIC3_BENCHMARK_PATH / "phenotyping"
MIMIC3_PHENO_24H_PATH = MIMIC3_BENCHMARK_PATH / "phenotyping_24h"

MIMIC4_BENCHMARK_PATH = DATA_PATH / "mimiciv_benchmark"
MIMIC4_RAW_PATH = DATA_PATH / "mimiciv"
MIMIC4_IHM_PATH = MIMIC4_BENCHMARK_PATH / "in-hospital-mortality"
MIMIC4_PHENO_PATH = MIMIC4_BENCHMARK_PATH / "phenotyping"
MIMIC4_PHENO_24H_PATH = MIMIC4_BENCHMARK_PATH / "phenotyping_24h"
MIMIC4_CXR_CSV_IHM = MIMIC4_BENCHMARK_PATH / "cxr/admission_w_cxr_ihm.csv"
MIMIC4_CXR_CSV_PHENO = MIMIC4_BENCHMARK_PATH / "cxr/admission_w_cxr_pheno.csv"

MIMIC_CXR_JPG_PATH = "/disk1/fywang/CXR_dataset/mimic_data/2.0.0/files"
# MIMIC_CXR_JPG_PATH = "/data1/r20user2/CXR_dataset/mimic_data/2.0.0/files"
