import numpy as np
import matplotlib.pyplot as plt
import h5py
import boto3
import shutil
import numpy as np
from os.path import isdir,isfile,join,exists
from os import mkdir,makedirs,getcwd,listdir

from data_utils import get_dataset_name, order_arranging, separate_list

root = "D:/Users/Lucas/Documents/MEG_Data_Cascade_Multiview/"

training_file_dir = root + "Data/train"
all_train_files = [f for f in listdir(training_file_dir) if isfile(join(training_file_dir, f))]
train_files_dirs = []
for i in range(len(all_train_files)):
    train_files_dirs.append(training_file_dir+'/'+all_train_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(train_files_dirs)
train_files_dirs = order_arranging(rest_list, mem_list, math_list, motor_list)

subjects = ['105923','164636','133019','113922','116726','140117','175237','177746','185442','191033','191437','192641']
list_subjects_test = ['204521','212318','162935','601127','725751','735148']

for subject in subjects:
    print("***" + str(subject))
    subject_files_train = []
    for item in train_files_dirs:
        if subject in item:
            subject_files_train.append(item)

    rest_matrix = np.random.rand(248,1)
    math_matrix = np.random.rand(248,1)
    memory_matrix = np.random.rand(248,1)
    motor_matrix = np.random.rand(248,1)

    for i in range(len(subject_files_train)):
        # with h5py.File(subject_files_train[i],'r') as f:
        #     dataset_name = get_dataset_name(subject_files_train[i])
        #     print(dataset_name)
        #     matrix = f.get(dataset_name)
        #     matrix = np.array(matrix)
        #     matrix = np.delete(matrix, [124], axis=0)
        #     matrix = np.delete(matrix, [235], axis=0)
        #     matrix = np.delete(matrix, [187], axis=0)
        if "rest" in subject_files_train[i]:
            with h5py.File(subject_files_train[i],'r') as f:
                dataset_name = get_dataset_name(subject_files_train[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
                # matrix = np.delete(matrix, [188, 236], axis=0)
            # assert matrix.shape[0] == 246 , "This rest data does not have {} channels, but {} instead".format('246',matrix.shape[0])
            rest_matrix = np.column_stack((rest_matrix, matrix))
            # for x in range(len(matrix)):
            #     print(x)
            #     print(matrix[x][0])

        if "math" in subject_files_train[i]:
            with h5py.File(subject_files_train[i],'r') as f:
                dataset_name = get_dataset_name(subject_files_train[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            # assert matrix.shape[0] == 246 , "This rest data does not have {} channels, but {} instead".format('246',matrix.shape[0])
            math_matrix = np.column_stack((math_matrix, matrix))
            
        if "memory" in subject_files_train[i]:
            with h5py.File(subject_files_train[i],'r') as f:
                dataset_name = get_dataset_name(subject_files_train[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            # assert matrix.shape[0] == 246 , "This rest data does not have {} channels, but {} instead".format('246',matrix.shape[0])
            memory_matrix = np.column_stack((memory_matrix, matrix))
            
        if "motor" in subject_files_train[i]:
            with h5py.File(subject_files_train[i],'r') as f:
                dataset_name = get_dataset_name(subject_files_train[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            # assert matrix.shape[0] == 246 , "This rest data does not have {} channels, but {} instead".format('246',matrix.shape[0])
            motor_matrix = np.column_stack((motor_matrix, matrix))

    np.save("./plot_data/rest_matrix.npy", rest_matrix)
    np.save("./plot_data/math_matrix.npy", math_matrix)
    np.save("./plot_data/memory_matrix.npy", memory_matrix)
    np.save("./plot_data/motor_matrix.npy", motor_matrix)


    rest_matrix = np.delete(rest_matrix, [0], axis=1)    
    data = np.swapaxes(rest_matrix, 0, 1)
    data = np.average(data, axis=0)
    outliers = abs(data - np.mean(data)) < 3 * np.std(data)
    channels = np.where(outliers == False)[0]
    print(channels)
    for channel in channels:
        rest_matrix[channels] = 0
    rest_matrix[124] = 0
    rest_matrix[236] = 0
    rest_matrix[188] = 0
    rest_matrix = np.swapaxes(rest_matrix, 0, 1)
    rest_matrix = rest_matrix[0::100]
    plt.plot(rest_matrix)
    plt.show()

    math_matrix = np.delete(math_matrix, [0], axis=1)
    data = np.swapaxes(math_matrix, 0, 1)
    data = np.average(data, axis=0)
    outliers = abs(data - np.mean(data)) < 3 * np.std(data)
    channels = np.where(outliers == False)[0]
    print(channels)
    for channel in channels:
        math_matrix[channels] = 0
    math_matrix = np.swapaxes(math_matrix, 0, 1)
    math_matrix = math_matrix[0::100]
    # plt.plot(math_matrix)
    # plt.show()

    motor_matrix = np.delete(motor_matrix, [0], axis=1)
    data = np.swapaxes(motor_matrix, 0, 1)
    data = np.average(data, axis=0)
    outliers = abs(data - np.mean(data)) < 3 * np.std(data)
    channels = np.where(outliers == False)[0]
    print(channels)
    for channel in channels:
        motor_matrix[channels] = 0
    motor_matrix = np.swapaxes(motor_matrix, 0, 1)
    motor_matrix = motor_matrix[0::100]
    # plt.plot(motor_matrix)
    # plt.show()


    memory_matrix = np.delete(memory_matrix, [0], axis=1)
    data = np.swapaxes(memory_matrix, 0, 1)
    data = np.average(data, axis=0)
    outliers = abs(data - np.mean(data)) < 3 * np.std(data)
    channels = np.where(outliers == False)[0]
    print(channels)
    for channel in channels:
        memory_matrix[channels] = 0
    memory_matrix = np.swapaxes(memory_matrix, 0, 1)
    memory_matrix = memory_matrix[0::100]
    # plt.plot(memory_matrix)
    # plt.show()