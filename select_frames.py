import csv
import os
import numpy as np

def compute_peak_frames(file_content, weight, index):

    # record every value of frames
    value_recorded = []

    for i in range(len(file_content)):

        # add to the value_recorded
        value_temp = 0

        for j in range(len(index)):
            if file_content[i,index[j]] != '9':
                value_temp +=  weight[j] * int(file_content[i,index[j]])

        value_recorded.append(value_temp)

    # record the max and select the maximium ones
    max_value = np.max(np.array(value_recorded))
    selected_file_name = file_content[np.where(np.array(value_recorded) == max_value)[0],0]

    return selected_file_name

# task1
weights_happiness = [0.82, 0.7, 0.57, 0.83, 0.63]
index_happiness = [12, 25, 6, 7, 10]
# task2
weights_sad = [0.53, 0.42, 0.31, 0.13, 0.1]
index_sad = [4, 15, 1, 7, 17]
# task3
weights_surprise = [0.38, 0.37, 0.85, 0.3, 0.5, 0.2]
index_surprise = [1, 2, 25, 26, 5, 7]
# task5
weights_fear = [0.52, 0.4, 0.85, 0.38, 0.57, 0.57]
index_fear = [1, 4, 25, 5, 7, 10]
# task7
weights_anger = [0.65, 0.45, 0.4, 0.33, 0.15]
index_anger = [4, 7, 25, 10, 9]
# task8
weights_disgust = [0.21, 0.85, 0.23, 0.6, 0.75, 0.8]
index_disgust = [9, 10, 17, 4, 7, 25]

root_path = 'AU_OCC'
output_path = 'selected_occ'
files = os.listdir(root_path)

# input_file = 'F003_T2.csv'
for input_file in files:
    
    # task name
    task_name = input_file.split('.')[0].split('_')[-1]

    if task_name == 'T4' or task_name == 'T6':
        continue
    else:
        file_path = os.path.join(root_path, input_file)
        file_name = input_file.split('.')[0]
        
        # store them in the list
        file_content = []
        csv_reader = csv.reader(open(file_path))
        for line in csv_reader:
            file_content.append(line)

        # change into array and delete the first line
        file_content = np.array(file_content)
        file_content = file_content[1:,:]

        if task_name == 'T1':
            selected_file_name = compute_peak_frames(file_content, weights_happiness, index_happiness)
        elif task_name == 'T2':
            selected_file_name = compute_peak_frames(file_content, weights_sad, index_sad)
        elif task_name == 'T3':
            selected_file_name = compute_peak_frames(file_content, weights_surprise, index_surprise)
        elif task_name == 'T5':
            selected_file_name = compute_peak_frames(file_content, weights_fear, index_fear)
        elif task_name == 'T7':
            selected_file_name = compute_peak_frames(file_content, weights_anger, index_anger)
        elif task_name == 'T8':
            selected_file_name = compute_peak_frames(file_content, weights_disgust, index_disgust)

        selected_file_name = np.array(selected_file_name, dtype=int)
        print(selected_file_name)
        np.savetxt(os.path.join(output_path, file_name + ".txt"), selected_file_name)
