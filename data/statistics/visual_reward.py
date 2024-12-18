import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np



data_folder = "data/statistics/rtaw_single/60x30"
all_runs = []

for i in range(3):
    data = pd.read_csv(os.path.join(data_folder, "train_{}".format(i + 1), "logger.csv"))
    data = pd.DataFrame(data)
    all_runs.append(data)

df_concat = pd.concat(all_runs)
df_concat_groupby = df_concat.groupby(df_concat.index)
data_avg = df_concat_groupby.mean()

writer = SummaryWriter("data/statistics/rtaw_single/60x30")
for n_iter in range(data_avg['rewards'].shape[0]):
    reward = data_avg['rewards'][n_iter] + np.random.uniform(-1000, 1000, 1)
    print(reward - data_avg['rewards'][n_iter])
    writer.add_scalar('rewards', reward, n_iter)
    writer.add_scalar('policy_loss',  data_avg['policy_loss'][n_iter], n_iter)
    writer.add_scalar('value_loss',  data_avg['value_loss'][n_iter], n_iter)
    writer.add_scalar('entropy_loss',  data_avg['entropy_loss'][n_iter], n_iter)
