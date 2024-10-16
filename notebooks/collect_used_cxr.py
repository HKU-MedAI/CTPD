import os
import shutil
import pandas as pd
from tqdm import tqdm
import ipdb


def main():
    filename = "/home/*/Documents/EHR_codebase/MMMSPG/data/mimiciv_benchmark/cxr/admission_w_cxr.csv"
    df = pd.read_csv(filename)
    df = df.dropna(subset=['dicom_id'])
    mimic_cxr_path = "/disk1/*/CXR_dataset/mimic_data/2.0.0/files"
    os.makedirs("/disk1/*/EHR_dataset/mimiciv_benchmark/used_cxrs", exist_ok=True)
    for row in tqdm(df.itertuples(), total=len(df)):
        img_path = os.path.join(mimic_cxr_path, row.path)
        assert os.path.exists(img_path), f"Image path {img_path} does not exist"
        img_name = img_path.split("/")[-1]
        shutil.copy(img_path, f"/disk1/*/EHR_dataset/mimiciv_benchmark/used_cxrs/{img_name}")


if __name__ == '__main__':
    main()