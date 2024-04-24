reinforce = np.load("reinforce.npy")
ac = np.load("ac.npy")
# reinforce = [np.load(f"reinforce{i}.npy") for i in range(10)]
# ac = [np.load(f"ac{i}.npy") for i in range(10)]
plt.rcParams.update({'font.size': 22})

r_avg = np.mean(reinforce, axis=1)
r_std = np.std(reinfoce, axis=1)
ac_avg = np.mean(ac, axis=1)
ac_std = np.std(ac, axis=1)

plt.figure(1, figsize=(10,7))
plt.plot(range(len(r_avg)), r_avg, color="#FF0000", label="REINFORCE reward")
plt.plot(range(len(ac_avg)), ac_avg, color="#0000FF", label="Actor Critic reward")
plt.fill_between(r_avg+r_std, r_avg-r_std, color="#FF000095")
plt.fill_between(ac_avg+ac_std, ac_avg-ac_std, color="#0000FF95")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Mean reward")
plt.savefig('rewards.png')
plt.show()

