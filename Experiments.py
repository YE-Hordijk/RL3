# Experiments.py - Ilse
# This script runs experiments with a given parameter set and stores the results
# in numpy files.

import numpy as np

import TestFunctions as RF # REINFORCE # TODO Vervangen door echte file
import TestFunctions as AC # ActorCritic # TODO Vervangen door echte file


#*******************************************************************************
def RunExperiment(ALGO, nameResults="testResults", repetition_count=20, params=None):
	multi_results = []
	for i in range(repetition_count):
		print(f"\033[42mRepetition {i+1}\033[0m", end='\n')
		# l = Learning(render_mode="rgb_array", **params) # initialize learning parameters # TODO Classe aanroepen die parameters initializeerd
		multi_results.append( ALGO.Algorithm(**params) )

	average_array = np.mean(multi_results, axis=0)
	std_dev_array = np.std(multi_results, axis=0)

	# Save average and std_dev in nupy array
	#np.savez(f'{nameResults}.npz', array1=average_array, array2=std_dev_array) # save results separate but in one zip
	
	print(average_array)
	print(std_dev_array)
	combinedResults = np.vstack((average_array, std_dev_array))
	np.save(f'{nameResults}', combinedResults) # save the results containing both average and stdev in one file
#*******************************************************************************


params = {'nrEpisodes': 100}
RunExperiment(RF, "rf.npy", repetition_count=20, params=params) # Do experiments for REINFORCE
RunExperiment(AC, "ac.npy", repetition_count=20, params=params) # Do experiments for Actor Critic





