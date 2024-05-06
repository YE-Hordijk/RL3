import numpy as np
import Plot_Results as PR


#*******************************************************************************
def sort_on_pivot(averages, pivot):
    config_list = configs.cons

    unique_pivot_values = set()
    for d in config_list:
        pivot_value = d.get(pivot)
        if pivot_value is not None:
            unique_pivot_values.add(pivot_value)

    pivot_dict = {pivot_value: [] for pivot_value in unique_pivot_values}

    for i, d in enumerate(config_list):
        pivot_dict[d[pivot]].append(averages[i])

    # Calculate averages
    average_dict = {}
    for key, lists_of_lists in pivot_dict.items():
        average_array = np.mean(lists_of_lists, axis=0)
        average_dict[key] = average_array

    print(average_dict)
    new_lists = []
    new_labels = []

    for key in average_dict:
        new_labels.append(f"{pivot}={key}")
        new_lists.append(average_dict[key])

    return new_lists, new_labels

#*******************************************************************************
#print(np.load(f'testFile_AC.npy')[1])
#exit()

# Load the NumPy file
averages = []
labels = []
label_values = [[0.01, 0.01],[0.01, 0.1],[0.01, 0.3],[0.001, 0.01],[0.001, 0.1],[0.001, 0.3],[0.0001, 0.01],[0.0001, 0.1],[0.0001, 0.3]]
stdevs = []
for i,j in zip(['rf_p_0.01_0.01', 'rf_p_0.01_0.1', 'rf_p_0.01_0.3','rf_p_0.001_0.01', 'rf_p_0.001_0.1', 'rf_p_0.001_0.3','rf_p_0.0001_0.01', 'rf_p_0.0001_0.1', 'rf_p_0.0001_0.3',], range(9)):
    averages.append( np.load(f'{i}.npy')[0] )
    labels.append(f'$\\alpha$: {label_values[j][0]}, $\epsilon$: {label_values[j][1]}')
    stdevs.append(np.load(f'{i}.npy')[1])

# Plot everything
PR.Plot_Multiple_Episode_Durations(averages, stdev_list=None, labels=labels, path='test.png', smoothen=True)










