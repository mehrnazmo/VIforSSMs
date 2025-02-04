import os
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def load_summaries(log_dir):
    # Initialize the event accumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Get the list of tags
    tags = ea.Tags()['scalars']

    # Initialize a dictionary to store the summaries
    summaries = {tag: [] for tag in tags}

    # Iterate through the tags and extract the values
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            summaries[tag].append((event.step, event.value))

    # Convert the summaries to NumPy arrays
    for tag in summaries:
        summaries[tag] = np.array(summaries[tag])

    return summaries

# Example usage
# # log_dir_parent = '/home/mmotame1/gmvi/ryder_run/VIforSSMs/locally_variant/train_fix_theta_sparse/series_0_10:10:24-00:47:29/'
# log_dir_parent = '/home/mmotame1/gmvi/ryder_run/VIforSSMs/locally_variant/train_fix_theta_sparse/series_0_10:10:24-02:11:30'
# # '/home/mmotame1/gmvi/ryder_run/VIforSSMs/locally_variant_last/train_sparse'
# summaries = load_summaries(log_dir_parent)
# print('summaries:', summaries.keys())
# time_arr = summaries['Elapsed Time']
# elbo = summaries['loss/ELBO']
# print('time_arr:', time_arr.shape)
# plt.plot(time_arr[:, 0], time_arr[:, 1])
# plt.title('Time')
# plt.savefig('time.png')
# plt.close()

# plt.figure()
# plt.plot(elbo[:, 0], elbo[:, 1])
# plt.title('ELBO')
# plt.savefig('elbo.png')
# plt.close()

# plt.figure()
# plt.plot(time_arr[1000:, 1]-time_arr[1000,1], elbo[:, 1])
# plt.title('Time vs ELBO')
# plt.savefig('time_vs_elbo.png')
# exit()
log_dir_parent = '/home/mmotame1/gmvi/ryder_run/VIforSSMs/locally_variant/train_fix_theta_sparse'

## filter to show only the series results
series = []
for d in sorted(os.listdir(log_dir_parent)):
    if d.startswith('series'):
        series.append(d)
print('series:', len(series))

all_log_dir = []       
for i in range(149):
    i_dir = []
    for d in series:
        if 'series_'+str(i) in d:
            i_dir.append(d)
    my_dir = i_dir[-1]
    log_dir = os.path.join(log_dir_parent, my_dir)
    all_log_dir.append(log_dir)
        
print('\n'*3)
print('log_dir:', len(all_log_dir))
print(all_log_dir[0])
all_steps = []
all_elbo = []
all_valid_elbo = []
all_times = []
all_valid_times = []
lengths = []
for i, log_dir in tqdm(enumerate(all_log_dir)):
    # print('log_dir:', i, log_dir)
    summaries = load_summaries(log_dir)
    # for tag, values in summaries.items():
    #     print(f"Tag: {tag}")
    values = summaries['loss/ELBO']
    Time_values = summaries['Elapsed Time']
    # print(Time_values.shape)
    l = values.shape[0]
    lengths.append(l)
    # print('values.shape:', values.shape)
    all_steps.append(values[:, 0])
    all_elbo.append(values[:, 1])  
    all_times.append(Time_values[:, 1])
    if l > 500:
        all_valid_elbo.append(values[:500, 1])
        if len(Time_values) < 1500:
            d = 1500 - len(Time_values)
            Time_values = np.concatenate([Time_values, np.zeros((d, 2))], axis=0)            
        all_valid_times.append(Time_values[1000:1500, 1])
    else:
        print('log_dir:', i, log_dir)
    # # Access the summaries as NumPy arrays
    # for tag, values in summaries.items():
    #     if tag == 'loss/ELBO':
    #         l = values.shape[0]
    #         lengths.append(l)
    #         # print('values.shape:', values.shape)
    #         all_steps.append(values[:, 0])
    #         all_elbo.append(values[:, 1])
    #     #     print(values.shape)
    #     # print(f"Tag: {tag}")
    #     # print(f"Steps: {values[:, 0]}")
    #     # print(f"Values: {values[:, 1]}")
        
valid_elbo_arr = np.array(all_valid_elbo)[:128,:]
print('valid_elbo_arr:', valid_elbo_arr.shape)
np.save('elbo_sparse_batch.npy', valid_elbo_arr)
exit()


elbo_mean = valid_elbo_arr.mean(axis=0)
print('elbo_mean:', elbo_mean.shape)
all_times_arr = np.array(all_times) ###################################################
print('all_times_arr:', all_times_arr.shape)
all_valid_times_arr = np.array(all_valid_times)
all_valid_times_arr[all_valid_times_arr == 0] = np.nan
all_valid_times_arr = np.nanmean(all_valid_times_arr, axis=0)
print('all_valid_times_arr:', all_valid_times_arr.shape)

exit()

np.save('elbo_mean_fix_theta_lv_dense_oct10.npy', elbo_mean)
np.save('all_times_arr_fix_theta_lv_dense_oct10.npy', all_times_arr)
np.save('all_valid_times_arr_fix_theta_lv_dense_oct10.npy', all_valid_times_arr)
np.save('all_steps_fix_theta_lv_dense_oct10.npy', all_steps)
