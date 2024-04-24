import numpy as np
from matplotlib import pyplot as plt

reinforce = np.load("rf.npy")
ac = np.load("ac.npy")
# reinforce = [np.load(f"reinforce{i}.npy") for i in range(10)]
# ac = [np.load(f"ac{i}.npy") for i in range(10)]
plt.rcParams.update({'font.size': 22})

r_avg = reinforce[0]
r_std = reinforce[1]
ac_avg = ac[0]
ac_std = ac[1]

plt.figure(1, figsize=(10,7))
plt.plot(range(len(r_avg)), r_avg, color="#FF0000", label="REINFORCE reward")
plt.plot(range(len(ac_avg)), ac_avg, color="#0000FF", label="Actor Critic reward")
plt.fill_between(range(len(r_avg), r_avg+r_std, r_avg-r_std, color="#FF000095")
plt.fill_between(range(len(ac_avg), ac_avg+ac_std, ac_avg-ac_std, color="#0000FF95")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Mean reward")
plt.savefig('rewards.png')
plt.show()

