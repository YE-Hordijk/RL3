import numpy as np
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

interval = 10
reinforce = np.load(f"rf_{str(interval)}.npy")
ac = np.load(f"ac_{str(interval)}.npy")

# reinforce = [np.load(f"reinforce{i}.npy") for i in range(10)]
# ac = [np.load(f"ac{i}.npy") for i in range(10)]
plt.rcParams.update({'font.size': 22})


r_avg = savgol_filter(reinforce[0], 3, 1)
r_std = reinforce[1]
ac_avg = savgol_filter(ac[0], 3, 1)
ac_std = ac[1]

# interval = params['interval']
def timesteps(arr, interval):
    return range(0, len(arr) * interval, interval)

plt.figure(1, figsize=(10,7))
plt.plot(timesteps(r_avg, interval), r_avg, color="#FF0000", label="REINFORCE reward")
plt.plot(timesteps(ac_avg, interval), ac_avg, color="#0000FF", label="Actor Critic reward")
plt.fill_between(timesteps(r_avg, interval), r_avg+r_std, r_avg-r_std, color="#FF000050")
plt.fill_between(timesteps(ac_avg, interval), ac_avg+ac_std, ac_avg-ac_std, color="#0000FF50")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Mean reward")
plt.savefig('rewards.png')
plt.show()
