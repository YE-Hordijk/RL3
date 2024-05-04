# Experiments.py - Ilse
# This script runs experiments with a given parameter set and stores the results
# in numpy files.

import numpy as np

from itertools import count, product
import REINFORCE as RF # REINFORCE
import ActorCritic as AC # ActorCritic

#*******************************************************************************
def RunExperiment(ALGO, nameResults="testResults", repetition_count=20, params=None):
	multi_results = []

	for i in range(repetition_count):
		print(f"\033[42mRepetition {i+1}\033[0m", end='\n')
		# l = Learning(render_mode="rgb_array", **params) # initialize learning parameters
		multi_results.append( ALGO.experiment(**params) )

	average_array = np.mean(multi_results, axis=0)
	std_dev_array = np.std(multi_results, axis=0)

	# Save average and std_dev in nupy array
	#np.savez(f'{nameResults}.npz', array1=average_array, array2=std_dev_array) # save results separate but in one zip
	
	print(average_array)
	print(std_dev_array)
	combinedResults = np.vstack((average_array, std_dev_array))

	np.save(f"{nameResults}_{params['interval']}.npy", combinedResults) # save the results containing both average and stdev in one file

	#np.save(f'{nameResults}', np.array([average_array, std_dev_array, params['interval']])) # save the results containing both average and stdev in one file
#*******************************************************************************

def RunParameters(ALGO, nameResults="testResults", repetition_count=20, params=None):
    print(params)
    for combination in product(*params.values()):
        param_set = dict(zip(params.keys(), combination))
        print(param_set)
        multi_results = []
        for i in range(repetition_count):
            print(f"\033[42mRepetition {i+1}\033[0m", end='\n')
		    # l = Learning(render_mode="rgb_array", **params) # initialize learning parameters
            multi_results.append( ALGO.parameters(**param_set) )

        average_array = np.mean(multi_results, axis=0)
        std_dev_array = np.std(multi_results, axis=0)

        # Save average and std_dev in nupy array
        #np.savez(f'{nameResults}.npz', array1=average_array, array2=std_dev_array) # save results separate but in one zip

        print(average_array)
        print(std_dev_array)
        combinedResults = np.vstack((average_array, std_dev_array))

        np.save(f"{nameResults}_{param_set['LearningRate']}_{param_set['epsilon']}.npy", combinedResults) # save the results containing both average and stdev in one file

#*******************************************************************************

params = {'nrEpisodes': 500,
		  'interval': 10,
		  'nrTestEpisodes': 5,
		 }
#RunExperiment(RF, "rf", repetition_count=5, params=params) # Do experiments for REINFORCE
parameters = {'nrEpisodes': [500],
		  'interval': [10],
		  'nrTestEpisodes': [5],
		  'LearningRate': [0.0001, 0.001, 0.01],
		  'epsilon': [0.01, 0.1, 0.3],
		 }
RunParameters(RF, "rf_p", repetition_count=5, params=parameters)
#RunExperiment(AC, "ac", repetition_count=5, params=params) # Do experiments for Actor Critic





