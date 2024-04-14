# RL2 plot stuff

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter

#*******************************************************************************

def Plot_Episode_Durations(scores_per_episodes, stdev=None, title='Result', path='test.png', smoothen=False):
    plt.figure(1)
    durations_t = torch.tensor(scores_per_episodes, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    plt.plot(durations_t.numpy())


    if smoothen:
        window_size = 10
        weights = torch.tensor([1.0] * window_size) / window_size
        smoothed = torch.conv1d(durations_t.view(1, 1, -1), weights.view(1, 1, -1), padding=(window_size - 1) // 2)[0, 0]
        plt.plot(smoothed.numpy(), color='orange', label='Smoothed')  # Plot smoothed data in orange


    if stdev is not None:
        plt.fill_between(range(len(durations_t)), durations_t - stdev, durations_t + stdev, color='lightgrey')  # Plot standard deviation in light grey


    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

#*******************************************************************************


# Plot multiple scores in one plot
def Plot_Multiple_Episode_Durations(scores_per_episodes_list, stdev_list=None, labels=None, title='Result', path='test.png', smoothen=False):
    plt.figure(1)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    
    print(scores_per_episodes_list)


    if not smoothen:
        for scores_per_episodes in scores_per_episodes_list:
            durations_t = torch.tensor(scores_per_episodes, dtype=torch.float)
            plt.plot(durations_t.numpy())

    if smoothen:
        smoothed_lines = []
        for scores_per_episodes in scores_per_episodes_list:
            smoothed_y = savgol_filter(scores_per_episodes, window_length=11, polyorder=3)
            smoothed_lines.append(smoothed_y)

        for i, smoothed_line in enumerate(smoothed_lines):
            if labels is not None:
                plt.plot(smoothed_line, label=f'{labels[i]}')
            else:
                plt.plot(smoothed_line, label=f'Smoothed Line {i+1}') 


    if stdev_list is not None:
        for durations_t, stdev in zip(scores_per_episodes_list, stdev_list):
            plt.fill_between(range(len(durations_t)), durations_t - stdev, durations_t + stdev, color='lightgrey')

    plt.legend()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()






