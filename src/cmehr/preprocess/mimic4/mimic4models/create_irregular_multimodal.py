import os
import argparse
import json
import pickle
import pandas as pd
import ipdb
from tqdm import tqdm
import numpy as np
import statistics as stat

from cmehr.preprocess.mimic4.mimic4models.readers import MultimodalReader, Reader
import cmehr.preprocess.mimic4.mimic4models.common_utils as common_utils
from cmehr.preprocess.mimic4.mimic4models.preprocessing import Discretizer
from cmehr.paths import *

'''
python -m mimic4models.create_irregular_multimodal --dataset_path /disk1/*/EHR_dataset/mimiciv_benchmark/cxr \
    --cxr_path /disk1/*/EHR_dataset/mimiciv_benchmark/cxr/admission_w_cxr.csv
'''
parser = argparse.ArgumentParser(
    description='Create irregular time series from MIMIC 4')
parser.add_argument("--output_dir", type=str,
                    default=ROOT_PATH / "output_mimic4")
parser.add_argument('--timestep', type=float, default=1.0,
                    help="fixed timestep used in the dataset")
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--small_part', dest='small_part', action='store_true')
parser.add_argument('--dataset_path', type=str, default="mimic4")
parser.add_argument('--cxr_path', type=str, 
                    default="/home/*/Documents/EHR_codebase/MMMSPG/data/mimiciv_fairness_benchmark/cxr/admission_w_cxr.csv")
args = parser.parse_args()


# TODO: check if this hyperparameter is useless.
args.period_length = 48
args.dataset_path = Path(args.dataset_path)
output_dir = args.output_dir / "self_supervised_multimodal"
os.makedirs(output_dir, exist_ok=True)
print(args)
channel_path = str(ROOT_PATH / "src/cmehr/preprocess/mimic4/mimic4models/resources/channel_info.json")
config_path = str(ROOT_PATH / "src/cmehr/preprocess/mimic4/mimic4models/resources/discretizer_config.json")


class Discretizer_multi(Discretizer):
    '''
    The same discretizer without one-hot encoding
    '''

    def __init__(self, timestep=0.8, store_masks=True, 
                 impute_strategy='zero', start_time='zero',
                 config_path=config_path,
                 channel_path=channel_path
                 ):
        super(Discretizer_multi, self).__init__(
            timestep, store_masks, impute_strategy, start_time, config_path)

        with open(channel_path) as f:
            self.channel_info = json.load(f)

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        data = np.zeros(shape=(N_bins, N_channels), dtype=float)
        # data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [
            ["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                value = self.channel_info[channel]['values'][value]
            data[bin_id, channel_id] = float(value)

        # binning
        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j])
                original_value[bin_id][channel_id] = row[j]

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    elif self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    else:
                        raise ValueError("impute strategy is invalid")

                    # write(data, bin_id, channel, imputed_value, begin_pos)
                    write(data, bin_id, channel, imputed_value)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(
                            original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    # write(data, bin_id, channel, imputed_value, begin_pos)
                    write(data, bin_id, channel, imputed_value)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :]))
                            for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []

        for channel in self._id_to_channel:
            new_header.append(channel)
        for channel in self._id_to_channel:
            new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)


def save_data(reader: Reader,
              discretizer: Discretizer_multi,
              outputdir: str,
              small_part: bool,
              mode: str,
              non_mask=None):

    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    irg_data = ret["X"]
    ts = ret["t"]
    dicom_ids = ret["dicom_ids"]
    names = ret["name"]
    # procesed by discretizer
    reg_data = [discretizer.transform(X, end=t)[0]
                for (X, t) in zip(irg_data, ts)]

    with open(os.path.join(outputdir, f"ts_{mode}.pkl"), 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump((irg_data, reg_data, dicom_ids, names), f)

    return


def extract_irregular(dataPath_in, dataPath_out, cxr_df):
    """ Extract irregular time series """

    # Opening JSON file
    channel_info_file = open(channel_path)
    # channel_info_file = open(str(
    #     ROOT_PATH / "src/cmehr/dataset/mimic4/mimic4models/resources/channel_info.json"))
    channel_info = json.load(channel_info_file)

    dis_config_file = open(config_path)
    # dis_config_file = open(str(
    #     ROOT_PATH / "src/cmehr/dataset/mimic4/mimic4models/resources/discretizer_config.json"))
    dis_config = json.load(dis_config_file)

    channel_name = dis_config['id_to_channel']
    is_catg = dis_config['is_categorical_channel']

    with open(dataPath_in, 'rb') as f:
        ireg_data, reg_data, dicom_ids, names = pickle.load(f)

    data_irregular = []
    for p_id, x, in tqdm(enumerate(ireg_data), total=len(ireg_data), desc="Extract irregular time series"):
        data_i = {}
        tt = []
        features_list = []
        features_mask_list = []
        # for each time series:
        for t_idx, feature in enumerate(x):
            f_list_per_t = []
            f_mask_per_t = []
            for f_idx, val in enumerate(feature):
                if f_idx == 0:
                    tt.append(round(float(val), 2))
                else:
                    head = channel_name[f_idx-1]
                    if val == '':
                        f_list_per_t.append(0)
                        f_mask_per_t.append(0)
                    else:
                        f_mask_per_t.append(1)
                        if is_catg[head]:
                            val = channel_info[head]['values'][val]
                        f_list_per_t.append(float(round(float(val), 2)))
            assert len(f_list_per_t) == len(f_mask_per_t)
            features_list.append(f_list_per_t)
            features_mask_list.append(f_mask_per_t)
        assert len(features_list) == len(features_mask_list) == len(tt)
        # reg_data: reg_data | mask
        data_i['reg_ts'] = reg_data[p_id]
        data_i['name'] = names[p_id]
        sub_df = cxr_df[cxr_df['dicom_id'].isin(dicom_ids[p_id])]
        sub_df = sub_df.sort_values(by='StudyDateTime')
        data_i['cxr_path'] = sub_df["path"].tolist()
        # compute the time difference between the study time and the intime
        time_diff = (pd.to_datetime(sub_df["StudyDateTime"]) - pd.to_datetime(sub_df["intime"])).values / np.timedelta64(1, 'h')
        data_i['cxr_time'] = time_diff.tolist()
        data_i['ts_tt'] = tt
        data_i['irg_ts'] = np.array(features_list)
        data_i['irg_ts_mask'] = np.array(features_mask_list)
        data_irregular.append(data_i)

    with open(dataPath_out, 'wb') as f:
        pickle.dump(data_irregular, f)

    channel_info_file.close()
    dis_config_file.close()


def mean_std(dataPath_in, dataPath_out):
    with open(dataPath_in, 'rb') as f:
        data = pickle.load(f)

    # feature dimension of irregular time series
    irg_f_num = data[0]['irg_ts'].shape[1]
    # feature dimension of regular time series
    reg_f_num = data[0]['reg_ts'].shape[1]
    irg_feature_list = [[] for _ in range(irg_f_num)]
    reg_feature_list = [[] for _ in range(reg_f_num)]

    for _, p_data in enumerate(data):
        irg_ts = p_data['irg_ts']
        irg_ts_mask = p_data['irg_ts_mask']
        reg_ts = p_data['reg_ts']

        for _, (ts, mask) in enumerate(zip(irg_ts, irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, mask)):
                # only keep observed data
                if mask_val == 1:
                    irg_feature_list[f_idx].append(val)

        for ts in reg_ts:
            for f_idx, (val, mask_val) in enumerate(zip(ts[:reg_f_num//2], ts[reg_f_num//2:])):
                # don't care about the mask
                reg_feature_list[f_idx].append(val)

    # note that only 15 clinical features are found in mimic iv.
    irg_means = []
    irg_stds = []
    reg_means = []
    reg_stds = []

    for idx, (irg_vals, reg_vals) in enumerate(zip(irg_feature_list, reg_feature_list)):
        irg_means.append(stat.mean(irg_vals))
        irg_stds.append(stat.stdev(irg_vals))
        reg_means.append(stat.mean(reg_vals))
        reg_stds.append(stat.stdev(reg_vals))

    with open(dataPath_out, 'wb') as f:
        pickle.dump((irg_means, irg_stds, reg_means, reg_stds), f)


def normalize(dataPath_in, dataPath_out, normalizer_path):
    ''' Normalize the irregular and regular time series '''
    with open(dataPath_in, 'rb') as f:
        data = pickle.load(f)

    with open(normalizer_path, 'rb') as f:
        irg_means, irg_stds, reg_means, reg_stds = pickle.load(f)

    for p_id, p_data in tqdm(enumerate(data), total=len(data), desc="Normalize the time series"):
        irg_ts = p_data['irg_ts']
        irg_ts_mask = p_data['irg_ts_mask']

        reg_ts = p_data['reg_ts']
        feature_dim = irg_ts.shape[1]
        for t_idx, (ts, ts_mask) in enumerate(zip(irg_ts, irg_ts_mask)):
            for f_idx, (val, mask_val) in enumerate(zip(ts, ts_mask)):
                # remove outlier
                if mask_val == 1:
                    normed_val = (val-irg_means[f_idx]) / irg_stds[f_idx]
                    # if z score is too large
                    if np.abs(normed_val) > 4:
                        normed_val = 0
                    irg_ts[t_idx][f_idx] = normed_val

        for t_idx, ts in enumerate(reg_ts):
            for f_idx, val in enumerate(ts[:feature_dim]):
                reg_ts[t_idx][f_idx] = (
                    val-reg_means[f_idx])/reg_stds[f_idx]

    with open(dataPath_out, 'wb') as f:
        pickle.dump(data, f)


def diff_float(time1, time2):
    # compute time2-time1
    # return differences in hours but as float
    h = (time2 - time1).astype('timedelta64[m]').astype(int)
    return h / 60.0


def create_irregular_ts():
    with open(config_path) as f:
        config = json.load(f)
    variables = config['id_to_channel']

    train_reader = MultimodalReader(
            dataset_dir=args.dataset_path / "train",
            listfile=args.dataset_path / "train_listfile.csv",
            period_length=args.period_length,
            columns=variables
        )
    val_reader = MultimodalReader(
            dataset_dir=args.dataset_path / "train",
            listfile=args.dataset_path / "val_listfile.csv",
            period_length=args.period_length,
            columns=variables
        )
    test_reader = MultimodalReader(
            dataset_dir=args.dataset_path / "test",
            listfile=args.dataset_path / "test_listfile.csv",
            period_length=args.period_length,
            columns=variables
        )

    discretizer = Discretizer_multi(timestep=float(args.timestep),
                                    store_masks=True,
                                    impute_strategy='previous',
                                    start_time='zero')

    # new header
    discretizer_header = discretizer.transform(
        train_reader.read_example(0)["X"])[1].split(',')

    cont_channels = [i for (i, x) in enumerate(
        discretizer_header) if x.find("->") == -1]

    print("Step 1: Load regular time series data")
    save_data(train_reader, discretizer, output_dir,
              args.small_part, mode='train')
    save_data(val_reader, discretizer, output_dir,
              args.small_part, mode='val')
    save_data(test_reader, discretizer, output_dir,
              args.small_part, mode='test')

    print("Step 2: Load irregular time series data")
    cxr_df = pd.read_csv(args.cxr_path)
    for mode in ['train', 'val', 'test']:
        extract_irregular(
            os.path.join(output_dir, f"ts_{mode}.pkl"),
            os.path.join(output_dir, f"ts_{mode}.pkl"),
            cxr_df
        )

    # calculate mean,std of ts
    print("Step 3: compute mean and std of the whole dataset")
    mean_std(
        os.path.join(output_dir, 'ts_train.pkl'),
        os.path.join(output_dir, 'mean_std.pkl')
    )

    print("Step 4: normalize the time series data")
    for mode in ['train', 'val', 'test']:
        normalize(
            os.path.join(output_dir, f"ts_{mode}.pkl"),
            os.path.join(output_dir, f"norm_ts_{mode}.pkl"),
            os.path.join(output_dir, 'mean_std.pkl')
        )


if __name__ == '__main__':
    create_irregular_ts()