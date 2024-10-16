import os
import pandas as pd
import numpy as np
import pickle

# TODO: fix this path later
mimic_iv_benchmark_path = "/disk1/*/EHR_dataset/mimiciv_benchmark"
split = "train"
test_starttime_path = os.path.join(mimic_iv_benchmark_path, split, f"{split}_starttime.pkl")
episodeToStartTimeMapping = {}


def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    return (a-b).astype('timedelta64[h]').astype(int)

path = os.path.join(mimic_iv_benchmark_path, split)
for findex, folder in enumerate(os.listdir(path)):
    # events_path = os.path.join(path, folder, 'events.csv')
    # events = pd.read_csv(events_path)

    if not folder.isdigit():
        continue

    stays_path = os.path.join(path, folder, 'stays.csv')
    stays_df = pd.read_csv(stays_path)
    hadm_ids = list(stays_df.hadm_id.values)
    intimes = stays_df.intime.values

    for ind, hid in enumerate(hadm_ids):
        # sliced = events[events.hadm_id == hid]
        # chart_times = sliced['charttime']
        # chart_times = chart_times.sort_values()

        # Start time is intime ...
        intime = intimes[ind]
        result = intime
        # # remove intime from charttime
        # result = -1
        # # pick the first charttime which is positive or > -eps (1e-6)
        # for t in chart_times:
        #     # compute t-intime in hours
        #     if diff(t, intime) > 1e-6:
        #         result = t
        #         break
        name = folder + '_' + str(ind+1)
        episodeToStartTimeMapping[name] = result

    if findex % 1000 == 0:
        print("Processed %d" % (findex + 1))

with open(test_starttime_path, 'wb') as f:
    pickle.dump(episodeToStartTimeMapping, f, pickle.HIGHEST_PROTOCOL)
