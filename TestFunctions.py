import random
import time
import sys

#*******************************************************************************
def Algorithm(nrEpisodes, interval):
	reward_per_episodes = []
	i = 0
	for ep in range(nrEpisodes):
		if i%interval==0: reward_per_episodes.append(random.randint(0, 10))
		i+=1

		progress_bar(ep+1, nrEpisodes) 
		time.sleep(0.005)
	return reward_per_episodes
#*******************************************************************************
def progress_bar(current, total, bar_length=20):
	percent = current / total
	progress_length = int(bar_length * percent)
	bar = '#' * progress_length + '-' * (bar_length - progress_length)
	sys.stdout.write(f'\r[{bar}] {percent:.0%}')
	sys.stdout.flush()
	if percent == 1.0: 
		sys.stdout.write('\r\033[K')
		sys.stdout.flush()
#*******************************************************************************
