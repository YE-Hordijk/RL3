import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

def timesteps(arr, interval):
    return range(0, len(arr) * interval, interval)

interval = 10
plot = 'x'
while plot not in 'yn':
    plot = input("Do you want to REINFORCE vs AC [Y], or something different [n]:").lower()
if plot == 'n':
    data = []
    labels = []
    while name := input("Filename:"):
        d = np.load(name)
        if len(d) == 2:
            data.append(d[0])
        else:
            data.append(d)
        labels.append(input("Plot label for " + name + ":"))
else:
    data = [
            np.load(f"rf_{str(interval)}.npy"),
            np.load(f"ac_{str(interval)}.npy")
    ]
    labels = [
            "REINFORCE reward",
            "Actor Critic reward"
    ]

plt.rcParams.update({'font.size': 22})
plt.figure(1, figsize=(10,7))

if plot == 'n':
    for i in range(len(data)):
        data[i] = savgol_filter(data[i], 10, 1)
    for i in range(len(data)):
        plt.plot(range(len(data[i])), data[i], label=labels[i])
else:
    data[0][0] = savgol_filter(data[0][0], 10, 1)
    data[1][0] = savgol_filter(data[1][0], 10, 1)
    plt.plot(timesteps(data[0][0], interval), data[0][0], color="#FF0000", label=labels[0])
    plt.plot(timesteps(data[1][0], interval), data[1][0], color="#0000FF", label=labels[1])
    plt.fill_between(timesteps(data[0][0], interval), data[0][0]+data[0][1], data[0][0]-data[0][1], color="#FF000050")
    plt.fill_between(timesteps(data[1][0], interval), data[1][0]+data[1][1], data[1][0]-data[1][1], color="#0000FF50")

plt.legend()
plt.title(input("Plot title:"))
plt.xlabel("Episodes")
plt.ylabel("Mean reward")
saveas = input("Save as (leave empty to not save):")
if saveas:
    plt.savefig(saveas)
plt.show()
