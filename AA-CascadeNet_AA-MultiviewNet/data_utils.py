import h5py
import boto3
import shutil
import numpy as np
from os.path import isdir,isfile,join,exists
from os import mkdir,makedirs,getcwd,listdir
import os
import mne
import reading_raw
import gc
from sklearn.utils import shuffle
from keras.utils import to_categorical
from multiprocessing import Pool
from scipy import stats
from mne.viz import plot_alignment, set_3d_title
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

def normalize(array):
    return stats.zscore(array)

#Given the number "n", it finds the closest that is divisible by "m"
#Used when splitting the matrices
def closestNumber(n, m) : 
    q = int(n / m)  
    n1 = m * q 
    if((n * m) > 0) : 
        n2 = (m * (q + 1))  
    else : 
        n2 = (m * (q - 1)) 
    if (abs(n - n1) < abs(n - n2)) : 
        return n1 
    return n2 

def get_raw_coordinates():
    subject = '140117'
    type_state='rest'
    hcp_path = os.getcwd()

    raw = reading_raw.read_raw(subject=subject, hcp_path=hcp_path, run_index=0, data_type=type_state)
    coords = [0] * 248
    for row in raw.info['chs']:
        name = row['ch_name']
        if([*name][0] =='A'):
            index = [*name][1:]
            coords[int("".join(index)) - 1] = (row['loc'][0:3])

    return coords
  
#Reads the binary file given the subject and the type of state
#If the reading is successful, returns the matrix of size (248,number_time_steps)
#If the reading is NOT successful, prints the problem and returns the boolean False
def get_raw_data(subject, type_state, hcp_path):
  try: #type of state for this subject might not exist
    print("Reading the binary file and returning the raw matrix ...")
    raw = reading_raw.read_raw(subject=subject, hcp_path=hcp_path, run_index=0, data_type=type_state)
    data = []
    label = []
    for row in raw.info['chs']:
        if(row['coord_frame'] != 0):
            data.append(row['loc'][0:3])
            label.append(row['ch_name'])
            if(row['scanno'] == 122):
                x1 = data[-1][0]
                y1 = data[-1][1]
                z1 = data[-1][2]
            if(row['scanno'] == 152):
                x2 = data[-1][0]
                y2 = data[-1][1]
                z2 = data[-1][2]
                dis = math.sqrt( math.pow(x1-x2, 2) + math.pow(y1-y2, 2) + math.pow(z1-z2, 2) )
                print(dis)
    X, Y, Z = zip(*data)
    # kwargs = dict(eeg=False, show_axes=True, verbose=True)
    # fig = plot_alignment(raw.info, meg=("helmet", "sensors", "ref"), **kwargs)
    # set_3d_title(figure=fig, title="Magnes 3600wh")

    ax = plt.axes(projection='3d')

    ax.scatter3D(X,Y,Z)
    for a in range(0, len(X)):
        ax.text(X[a], Y[a], Z[a], label[a])
    plt.show()

    raw.load_data()
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    raw_matrix = raw[meg_picks[:]][0]
    del raw
    return raw_matrix
  except Exception as e:
    print("Problem in reading the file: The type of state '{}' might not be there for subject '{}'".format(type_state, subject))
    print("Exception error : ",e)
    return False

def create_data_directory():
    print("Creating data directory or skipping if already existing...")
    if(not isdir("Data")):
        try:
            mkdir("Data") 
            print("Created Data folder !")
        except Exception as e:
            print ("Creation of the Data directory failed")
            print("Exception error: ",str(e))
            
    if(not isdir("Data/train")):
        try:
            mkdir("Data/train")    
            print("Created train folder !")
        except Exception as e:
            print ("Creation of the train directory failed")
            print("Exception error: ",str(e))
            
    if(not isdir("Data/validate")):
        try:
            mkdir("Data/validate")  
            print("Created validate folder !")
        except Exception as e:
            print ("Creation of the validate directory failed")
            print("Exception error: ",str(e))
            
    if(not isdir("Data/test")):
        try:
            mkdir("Data/test")    
            print("Created test folder !")
        except Exception as e:
            print ("Creation of the test directory failed")
            print("Exception error: ",str(e))

def create_h5_files(raw_matrix,subject,type_state):
    print()
    print("shape of raw matrix",raw_matrix.shape)
    print()
    train_folder = "Data/train/"
    validate_folder = "Data/validate/"
    test_folder = "Data/test/"
    
    number_epochs = 250
    time_steps_per_epoch = 1425
    number_columns = number_epochs * time_steps_per_epoch

    number_columns_per_chunk = number_columns // 10

    if subject == "212318" or subject == "162935" or subject == "204521" or subject == "601127" or subject == "725751" or subject == "735148":  
        #data goes to test folder
        for i in range(10):
            start_index_col = number_columns_per_chunk * (i+4) # i+4 corresponds to an offset of 30s (approximately) from  the start
            stop_index_col = start_index_col + number_columns_per_chunk - 1 
            destination_file = test_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
            with h5py.File(destination_file, "w") as hf:
                    hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ])       
        
    else:
        #data goes to train and validate folder
        for i in range(10):
            if i >= 0 and i < 8:
                destination_file = train_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
            if i >= 8 and i < 10:
                destination_file = validate_folder + type_state+'_'+subject+'_'+str(i+1)+'.h5'
            start_index_col = number_columns_per_chunk * (i+4) # i+4 corresponds to an offset of 30s (approximately) from  the start
            stop_index_col = start_index_col + number_columns_per_chunk - 1
            with h5py.File(destination_file, "w") as hf:
                hf.create_dataset(type_state+'_'+subject, data=raw_matrix[ : , start_index_col : stop_index_col ])      

##For each subject, it prints how many rest files and how many task files it has in the Amazon server            
def get_info_files_subjects(personal_access_key_id,secret_access_key, subjects):
    folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
    s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
    for subject in subjects:
        rest_count = 0
        task_count = 0
        for folder in folders:
            number_files = s3.list_objects_v2(Bucket="hcp-openaccess", Prefix='HCP_1200/'+subject+'/unprocessed/MEG/'+folder)['KeyCount']
            if "Restin" in folder:
                rest_count += number_files
            else:
                task_count += number_files
        print("for subject {}, rest_count = {}, and task_count = {}".format(subject, rest_count, task_count))
        
#Gets a list of subjects and returns a new list of subjects that might contain less subjects
#It iterates through all the subjects, and if a subject has 0 task or rest state files, it discards the subject
#Used to fix data imbalance and to prevent bugs during the training using the data generator
def get_filtered_subjects(personal_access_key_id,secret_access_key, subjects):
    print("starting to discard subjects to keep data balance for multi class classification...")
    print("\nA subject that contains every data should have 12 files for rest and 36 files for task\n")
    new_subject_list = []
    number_subjects = len(subjects)
    folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
    s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
    for subject in subjects:
        rest_count = 0
        task_count = 0
        for folder in folders:
            number_files = s3.list_objects_v2(Bucket="hcp-openaccess", Prefix='HCP_1200/'+subject+'/unprocessed/MEG/'+folder)['KeyCount']
            if "Restin" in folder:
                rest_count += number_files
            else:
                task_count += number_files
        if (rest_count == 12 and task_count == 36):
            new_subject_list.append(subject)
        else:
            print("Discarding subject '{}' because it had {} rest files and {} task files".format(subject, rest_count, task_count))


    new_list_len = len(new_subject_list)
    print("-"*7 + " Done filtering out subjects ! " + "-"*7)
    print("Original list had {} subjects and new list has {} subjects".format(number_subjects, new_list_len))
    return new_subject_list

###Downloads 1 subject and ignores the case where 1 of the folders/files is missing    
def download_subject(subject,personal_access_key_id,secret_access_key):
  s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
  #folders = ["3-Restin"]
  folders = ["3-Restin","4-Restin","5-Restin","6-Wrkmem","7-Wrkmem","8-StoryM","9-StoryM","10-Motort","11-Motort"]
  filenames = ["c,rfDC", "config", "e,rfhp1.0Hz,COH", "e,rfhp1.0Hz,COH1"]

  print("Creating the directories for the subject '{}'".format(subject))
  print()
  if exists(getcwd()+"//"+subject) == False:
    for folder in folders:
        makedirs(subject+"/unprocessed/MEG/"+folder+"/4D/")
  print("done !")
  print()
  print("Will start downloading the following files for all folders:")
  print(filenames)
  print()
  print()
  for filename in filenames:
    for folder in folders:
      if filename == "c,rfDC":
        print("downloading c,rfDC file for folder {} ...".format(folder))
        print()
      if(exists(getcwd()+"//"+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)):
        print("File already exists, moving on ...")
        print()
        pass
      try:
        s3.download_file('hcp-openaccess', 'HCP_1200/'+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename, subject+'/unprocessed/MEG/'+folder+'/4D/'+filename)
        if filename == "c,rfDC":
          print("done downloading c,rfDC for folder {} !".format(folder))
          print()
      except Exception as e:
        print()
        print("the folder '{}' for subject '{}' does not exist in Amazon server, moving to next folder ...".format(folder,subject))
        print("Exception error message: "+str(e))
        pass
    
#Main function to be executed to download subjects
#list_subjects should be a list of strings containing the 6 digits subjects
#hcp_path should be the current working directory (os.getcwd())
def download_batch_subjects(list_subjects, personal_access_key_id, secret_access_key, hcp_path): # hcp_path should be os.getcwd()
  create_data_directory()  
  state_types = ["rest", "task_working_memory", "task_story_math", "task_motor"]
  for subject in list_subjects:
    download_subject(subject,personal_access_key_id,secret_access_key)
    for state in state_types:
      matrix_raw = get_raw_data(subject, state, hcp_path)
      if type(matrix_raw) != type(False): # if the reading was done successfully
        print()
        print("Creating the uncompressed h5 files ...")
        create_h5_files(matrix_raw,subject,state)
    print("done creating the uncompressed h5 files for subject '{}' !".format(subject))
    print()
    print("deleting the directory containing the binary files of subject '{}' ...".format(subject))
    print()
    # try:
    #   shutil.rmtree(subject+"/",ignore_errors=True)#Removes the folder and all folders/files inside
    #   print("Done deleting the directory of the binary files!")
    #   print("Moving on to the next subject ...")
    #   print()
    # except Exception as e :
    #   print()
    #   print("Error while trying to delete the directory.")
    #   print("Exception message : " + str(e))
      
def separate_list(all_files_list):
    rest_list = []
    mem_list = []
    math_list = []
    motor_list = []
    for item in all_files_list:
        if "rest" in item:
            rest_list.append(item)
        if "memory" in item:
            mem_list.append(item)
        if "math" in item:
            math_list.append(item)
        if "motor" in item:
            motor_list.append(item)            
    return rest_list, mem_list, math_list, motor_list

def order_arranging(rest_list,mem_list,math_list,motor_list):
    ordered_list = []
    for index, (value1, value2, value3, value4) in enumerate(zip(rest_list, mem_list, math_list, motor_list)):
        ordered_list.append(value1)
        ordered_list.append(value2)
        ordered_list.append(value3)
        ordered_list.append(value4)
    return ordered_list

def multi_processing_cascade(directory, length, num_cpu,depth):
    assert len(directory) == length*num_cpu,"Directory does not have {} files.".format(length*num_cpu)
    window_size = 10
    input_rows = 20
    input_columns = 21
    split = []

    for i in range(num_cpu):
        split.append(directory[i*length:(i*length)+length])
        
    for i in range(len(split)):
        split[i] = (split[i],depth)

    pool = Pool(num_cpu)
    results = pool.map(load_overlapped_data_cascade, split)
    pool.terminate()
    pool.join()
    
    y = np.random.rand(1,4)
    for i in range(len(results)):
        y = np.concatenate((y,results[i][1]))

    y = np.delete(y,0,0)
    gc.collect()
    
    x={}

    x_temp = np.random.rand(1,input_rows,input_columns,depth)
    for i in range(window_size):
        for j in range(len(results)):
            x_temp = np.concatenate((x_temp,results[j][0]["input"+str(i+1)]))
        x_temp= np.delete(x_temp,0,0)
        x["input"+str(i+1)] = x_temp
        x_temp = np.random.rand(1,input_rows,input_columns,depth)
        gc.collect()
    return x, y

def multi_processing_multiviewGAT(directory, length, num_cpu,depth):
    assert len(directory) == length*num_cpu,"Directory does not have {} files.".format(length*num_cpu)
    input_channels = 248
    window_size = 10
    results = load_overlapped_data_multieviewGAT(directory, depth)

    y = results[1]
    gc.collect()
    
    x={}

    x_input = np.random.rand(1,input_channels,depth)
    for i in range(window_size):
        x_input = np.concatenate((x_input,results[0]["input"+str(i+1)]))
        x_input= np.delete(x_input,0,0)
        x["input"+str(i+1)] = x_input
        x_input = np.random.rand(1,input_channels,depth)
        gc.collect()

    return x, y
  
def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def preprocess_data_type_original(matrix, window_size,depth):
    input_rows = 20
    input_columns = 21
    input_channels = 248

    if(matrix.shape[1] == 1):
        length = 1
    else:
        length = closestNumber(int(matrix.shape[1]) - window_size*depth,window_size*depth)

    meshes = np.zeros((input_rows,input_columns,length),dtype=np.float64)
    for i in range(length):
        print(f"{i}/{length}", end='\r')
        array_time_step = np.reshape(matrix[:,i],(1,input_channels))
        meshes[:,:,i] = array_to_mesh(array_time_step)
    
    del matrix    

    inputs = []
    if length == 1:
        for i in range(window_size):
            inputs.append(np.zeros((0,input_rows,input_columns,depth)))
    else:
        column_offset = int(window_size*depth/2)    # difference between values in columns
        num_rows_big_matrix = int((length-window_size*depth/2)/column_offset) # number of rows

        for j in range(num_rows_big_matrix):
            if j == 0:
                for i in range(window_size):
                    inputs.append(np.zeros((num_rows_big_matrix,input_rows,input_columns,depth)))
                    inputs[i][j] = meshes[:,:,i*depth:(i+1)*depth]
            else:
                for i in range(window_size):
                    inputs[i][j] = meshes[:,:,column_offset*j+i*depth:column_offset*j+(i+1)*depth]

    del meshes
    gc.collect()
    
    number_y_labels = int((length/(window_size*depth)*2)-1)
    y = np.ones((number_y_labels,1),dtype=np.int8)
    return inputs, y

def preprocess_data_type(matrix,window_size,depth,):
    input_channels = 248

    if(matrix.shape[1] == 1):
        length = 1
    else:
        length = closestNumber(int(matrix.shape[1]) - window_size*depth, window_size*depth)
    matrices = np.zeros((input_channels,length),dtype=np.float64)
    for i in range(length):
        print(f"\t {i}/{length}", end='\r')
        matrix_step=np.reshape(matrix[:,i],(1,input_channels))
        matrices[:,i] = matrix_step
    print("\t [complete] \t \t")
    del matrix

    inputs = []
    if length == 1:
        for i in range(window_size):
            inputs.append(np.zeros((0,input_channels,depth)))
    else:
        overlap = 0.5 
        var = int(window_size*depth*overlap) # difference between values in columns
        var2 = int((length-window_size*depth*overlap)/var) # number of rows

        for j in range(var2):
            if j == 0:
                for i in range(window_size):
                    inputs.append(np.zeros((var2,input_channels,depth)))
                    inputs[i][j] = matrices[:,i*depth:(i+1)*depth]
            else:
                for i in range(window_size):
                    inputs[i][j] = matrices[:,var*j+i*depth:var*j+(i+1)*depth]
    del matrices
    gc.collect()

    y = np.ones((var2,1),dtype=np.int8)
    return inputs, y

def reshape_input_dictionary(input_dict, output_list, batch_size,depth):
    input_channels = 248
    length_training = output_list.shape[0]
    length_adapted_batch_size= closestNumber(length_training-batch_size,batch_size)
    

    for i in range(len(input_dict.keys())):
        input_dict["input"+str(i+1)] = np.reshape(input_dict["input"+str(i+1)][0:length_adapted_batch_size],(length_adapted_batch_size,input_channels,depth))
        # input_dict["input"+str(i+1)][:,124] = [0]*10

    output_list = output_list[0:length_adapted_batch_size]
    return input_dict, output_list

def load_overlapped_data_cascade(file_dirs_depth):
    
    input_rows = 20
    input_columns = 21
    input_channels = 248 #MEG channels
    number_classes = 4
    window_size = 10
    depth = file_dirs_depth[1]

    rest_matrix = np.random.rand(input_channels,1)
    math_matrix = np.random.rand(input_channels,1)
    memory_matrix = np.random.rand(input_channels,1)
    motor_matrix = np.random.rand(input_channels,1)
 

    files_to_load = file_dirs_depth[0]
    
    for i in range(len(files_to_load)):
        if "rest" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This rest data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            rest_matrix = np.column_stack((rest_matrix, matrix))

        if "math" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This math data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            math_matrix = np.column_stack((math_matrix, matrix))
            
        if "memory" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This memory data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            memory_matrix = np.column_stack((memory_matrix, matrix))
            
        if "motor" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This motor data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            motor_matrix = np.column_stack((motor_matrix, matrix))
        matrix = None

    x_rest,y_rest = preprocess_data_type(rest_matrix, window_size,depth) 
    rest_matrix = None
    y_rest = y_rest*0

    x_math,y_math = preprocess_data_type(math_matrix,window_size,depth)
    math_matrix = None
    gc.collect()

    x_mem,y_mem = preprocess_data_type(memory_matrix,window_size,depth)
    memory_matrix = None
    y_mem = y_mem * 2
   
    x_motor,y_motor = preprocess_data_type(motor_matrix,window_size,depth)
    motor_matrix = None
    y_motor = y_motor * 3
    gc.collect()
       
    dict_list = []
    for i in range(window_size):
        dict_list.append({0:x_rest[i], 1:x_math[i], 2:x_mem[i], 3:x_motor[i]})
        x_rest[i]= None
        x_math[i]= None
        x_mem[i]= None
        x_motor[i]= None
        gc.collect()
 
    inputs = []
    for i in range(window_size):
        inputs.append(np.random.rand(1,input_rows,input_columns,depth))

    for i in range(number_classes):
        for j in range(window_size):
            if dict_list[j][i].shape[0]>0:
                inputs[j]=np.concatenate((inputs[j],dict_list[j][i]))
                
    dict_list = None
    gc.collect()
    
    for i in range(window_size):
        inputs[i] = np.delete(inputs[i],0,0)
    
    dict_y = {0:y_rest,1:y_math,2:y_mem,3:y_motor}
    
    y = np.random.rand(1,1)
    for i in range(number_classes):
        if dict_y[i].shape[0]>0:
            y = np.concatenate((y,dict_y[i]))


    y = np.delete(y,0,0)

    # inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9],y = shuffle(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9],y,random_state=42)
    *inputs,y = shuffle(*inputs,y,random_state=42)
    x_length = inputs[0].shape[0]
    for i in range(x_length):
        for j in range(window_size):
            temp = inputs[j][i] # length,rows,columns,depth
            for k in range(depth):
                inside = temp[:,:,k]
                norm = normalize(inside)
                inputs[j][i][:,:,k] = norm

    temp = None
    inside = None
    norm = None
    gc.collect()

    data_dict = {'input1' : inputs[0], 'input2' : inputs[1],'input3' : inputs[2], 'input4': inputs[3], 'input5' : inputs[4],
                 'input6' : inputs[5], 'input7' : inputs[6],'input8' : inputs[7], 'input9': inputs[8], 'input10' : inputs[9]}
    
    inputs = None
    gc.collect()
    y = to_categorical(y,number_classes)

    return data_dict,y

def load_overlapped_data_multieviewGAT(file_dirs, depth):
    input_channels = 248 #MEG channels
    number_classes = 4
    window_size = 10

    rest_matrix = np.random.rand(input_channels,1)
    math_matrix = np.random.rand(input_channels,1)
    memory_matrix = np.random.rand(input_channels,1)
    motor_matrix = np.random.rand(input_channels,1)

    files_to_load = file_dirs
    print()
    print("Loading files")
    for i in range(len(files_to_load)):
        print(f"\t {i+1}/{len(files_to_load)}", end='\r')
        if "rest" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This rest data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            rest_matrix = np.column_stack((rest_matrix, matrix))

        if "math" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This math data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            math_matrix = np.column_stack((math_matrix, matrix))
            
        if "memory" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This memory data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            memory_matrix = np.column_stack((memory_matrix, matrix))
            
        if "motor" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This motor data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            motor_matrix = np.column_stack((motor_matrix, matrix))

        matrix = None
    print("")
    print("Preprocessing resting")
    x_rest_lstm,y_rest_lstm = preprocess_data_type(rest_matrix,window_size,depth)  
    rest_matrix = None
    y_rest_lstm = y_rest_lstm*0

    print("Preprocessing math task")
    x_math_lstm,y_math_lstm = preprocess_data_type(math_matrix,window_size,depth)       
    math_matrix = None
    
    print("Preprocessing memory task")
    x_mem_lstm,y_mem_lstm = preprocess_data_type(memory_matrix,window_size,depth)
    memory_matrix = None
    y_mem_lstm = y_mem_lstm*2
    
    print("Preprocessing motor task")
    x_motor_lstm,y_motor_lstm = preprocess_data_type(motor_matrix,window_size,depth)
    motor_matrix = None
    y_motor_lstm = y_motor_lstm * 3
    dicts_input = []
    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        dicts_input.append({0:x_rest_lstm[i], 1:x_math_lstm[i], 2:x_mem_lstm[i], 3:x_motor_lstm[i]})
        x_rest_lstm[i]= None
        x_math_lstm[i]= None
        x_mem_lstm[i]= None
        x_motor_lstm[i]= None
        gc.collect()

    inputs = []

    # Create [window_size] empty windows
    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        inputs.append(np.random.rand(1,input_channels,depth))

    #Combine 4 class-seperated windows to one single window
    for i in range(number_classes):
        for j in range(window_size):
            print(f"Combining classes [{i+1}] {j+1}/{window_size} \t", end='\r')
            if dicts_input[j][i].shape[0]>0:
                inputs[j]=np.concatenate((inputs[j],dicts_input[j][i]))
                dicts_input[j][i] = None
            gc.collect()
    
    print("")
    
    #Delete empty placeholders
    for i in range(window_size):
        inputs[i] = np.delete(inputs[i],0,0)

    dict_y = {0:y_rest_lstm,1:y_math_lstm,2:y_mem_lstm,3:y_motor_lstm}

    #Combine 4 class-seperated windows to one single window
    y = np.random.rand(1,1)
    for i in range(number_classes):
        if dict_y[i].shape[0]>0:
            y = np.concatenate((y,dict_y[i]))
    y = np.delete(y,0,0)

    # inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9],inputs[10],inputs[11],inputs[12],inputs[13],inputs[14],inputs[15],inputs[16],inputs[17],inputs[18],inputs[19],y = shuffle(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9], inputs[10],inputs[11],inputs[12],inputs[13],inputs[14],inputs[15],inputs[16],inputs[17],inputs[18],inputs[19], y,random_state=42)
    print("Shuffling...", end='\r')
    *inputs,y = shuffle(*inputs,y,random_state=42)
    print("Shuffling finished")
    
    print("Shuffling...", end='\r')
    x_length = inputs[0].shape[0]
    for i in range(x_length):
        print(f"Normalizing {i} / {x_length} ", end='\r')
        for j in range(window_size):
            for k in range(depth):
                temp = inputs[j][i]
                inside = temp[:,k]
                norm = normalize(inside)
                inputs[j][i][:,k] = norm
    print("Normalizing finished \t\t")
    
    del temp
    del inside
    del norm
    gc.collect()
    
    assert len(inputs) == 10, "inputs does not have size 10, but has size {}".format(len(inputs)) 
    data_dict = {'input1' : inputs[0], 'input2' : inputs[1],'input3' : inputs[2], 'input4': inputs[3], 'input5' : inputs[4],
                 'input6' : inputs[5], 'input7' : inputs[6],'input8' : inputs[7], 'input9': inputs[8], 'input10' : inputs[9]}
    
    del inputs
    gc.collect()
    
    y = to_categorical(y,number_classes)
    return data_dict,y

def load_overlapped_data_multiview(file_dirs_depth):

    input_rows = 20
    input_columns = 21
    input_channels = 248 #MEG channels
    number_classes = 4
    window_size = 10
    depth = file_dirs_depth[1]

    rest_matrix = np.random.rand(input_channels,1)
    math_matrix = np.random.rand(input_channels,1)
    memory_matrix = np.random.rand(input_channels,1)
    motor_matrix = np.random.rand(input_channels,1)

    files_to_load = file_dirs_depth[0]

    for i in range(len(files_to_load)):
        print(f"{i}/{len(files_to_load)}")
        if "rest" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This rest data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            rest_matrix = np.column_stack((rest_matrix, matrix))

        if "math" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This math data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            math_matrix = np.column_stack((math_matrix, matrix))
            
        if "memory" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This memory data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            memory_matrix = np.column_stack((memory_matrix, matrix))
            
        if "motor" in files_to_load[i]:
            with h5py.File(files_to_load[i],'r') as f:
                dataset_name = get_dataset_name(files_to_load[i])
                matrix = f.get(dataset_name)
                matrix = np.array(matrix)
            assert matrix.shape[0] == input_channels , "This motor data does not have {} channels, but {} instead".format(input_channels,matrix.shape[0])
            motor_matrix = np.column_stack((motor_matrix, matrix))

        matrix = None

    print("rest")
    x_rest,y_rest = preprocess_data_type(rest_matrix,window_size,depth)
    x_rest_lstm,y_rest_lstm = preprocess_data_type_lstm(rest_matrix,window_size,depth)  
    rest_matrix = None
    y_rest = y_rest*0
    y_rest_lstm = y_rest_lstm*0

    print("")
    print("math")
    x_math,y_math = preprocess_data_type(math_matrix,window_size,depth)
    x_math_lstm,y_math_lstm = preprocess_data_type_lstm(math_matrix,window_size,depth)       
    math_matrix = None
    
    print("")
    print("mem")
    x_mem,y_mem = preprocess_data_type(memory_matrix,window_size,depth)
    x_mem_lstm,y_mem_lstm = preprocess_data_type_lstm(memory_matrix,window_size,depth)
    memory_matrix = None
    y_mem = y_mem*2
    y_mem_lstm = y_mem_lstm*2
    
    print("")
    print("motor")
    x_motor,y_motor = preprocess_data_type(motor_matrix,window_size,depth)
    x_motor_lstm,y_motor_lstm = preprocess_data_type_lstm(motor_matrix,window_size,depth)
    motor_matrix = None
    y_motor = y_motor * 3
    y_motor_lstm = y_motor_lstm * 3
    
    dicts_cnn = []
    dicts_lstm = []
    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        dicts_cnn.append({0:x_rest[i], 1:x_math[i], 2:x_mem[i], 3:x_motor[i]})
        dicts_lstm.append({0:x_rest_lstm[i], 1:x_math_lstm[i], 2:x_mem_lstm[i], 3:x_motor_lstm[i]})
        x_rest[i]= None
        x_math[i]= None
        x_mem[i]= None
        x_motor[i]= None
        x_rest_lstm[i]= None
        x_math_lstm[i]= None
        x_mem_lstm[i]= None
        x_motor_lstm[i]= None
        gc.collect()

    inputs = []
    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        inputs.append(np.random.rand(1,input_rows,input_columns,depth))

    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        inputs.append(np.random.rand(1,input_channels,depth))

    for i in range(number_classes):
        for j in range(window_size):
            print(f"{j}/{window_size}", end='\r')
            if dicts_cnn[j][i].shape[0]>0:
                inputs[j]=np.concatenate((inputs[j],dicts_cnn[j][i]))
                dicts_cnn[j][i] = None
            if dicts_lstm[j][i].shape[0]>0:
                inputs[j+window_size]=np.concatenate((inputs[j+window_size],dicts_lstm[j][i]))
                dicts_lstm[j][i] = None
            gc.collect()
        
    for i in range(window_size):
        print(f"{i}/{window_size}", end='\r')
        inputs[i] = np.delete(inputs[i],0,0)
        inputs[i+window_size] = np.delete(inputs[i+window_size],0,0)
        
    dict_y = {0:y_rest,1:y_math,2:y_mem,3:y_motor}
    
    
    y = np.random.rand(1,1)
    for i in range(number_classes):
        if dict_y[i].shape[0]>0:
            y = np.concatenate((y,dict_y[i]))

    y = np.delete(y,0,0)

    # inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9],inputs[10],inputs[11],inputs[12],inputs[13],inputs[14],inputs[15],inputs[16],inputs[17],inputs[18],inputs[19],y = shuffle(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9], inputs[10],inputs[11],inputs[12],inputs[13],inputs[14],inputs[15],inputs[16],inputs[17],inputs[18],inputs[19], y,random_state=42)
    print("Shuffling...", end='\r')
    *inputs,y = shuffle(*inputs,y,random_state=42)
    print(f"Shuffling done", end='\r')
    x_length = inputs[0].shape[0]
    for i in range(x_length):
        for j in range(window_size):
            for k in range(depth):
                temp = inputs[j][i]
                inside = temp[:,:,k]
                norm = normalize(inside)
                inputs[j][i][:,:,k] = norm

                temp = inputs[j+window_size][i]
                inside = temp[:,k]
                norm = normalize(inside)
                inputs[j+window_size][i][:,k] = norm

    del temp
    del inside
    del norm
    gc.collect()
    
    data_dict = {'input1' : inputs[0], 'input2' : inputs[1],'input3' : inputs[2], 'input4': inputs[3], 'input5' : inputs[4],
                 'input6' : inputs[5], 'input7' : inputs[6],'input8' : inputs[7], 'input9': inputs[8], 'input10' : inputs[9],
                 'input11' : inputs[10], 'input12' : inputs[11],'input13' : inputs[12], 'input14': inputs[13], 'input15' : inputs[14],
                 'input16' : inputs[15], 'input17' : inputs[16],'input18' : inputs[17], 'input19': inputs[18], 'input20' : inputs[19]}
    
    del inputs
    gc.collect()
    
    y = to_categorical(y,number_classes)
    return data_dict,y

root = "D:/Users/Lucas/Documents/MEG_Data_Cascade_Multiview/"

training_file_dir = root + "Data/train"
all_train_files = [f for f in listdir(training_file_dir) if isfile(join(training_file_dir, f))]
train_files_dirs = []
for i in range(len(all_train_files)):
    train_files_dirs.append(training_file_dir+'/'+all_train_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(train_files_dirs)
train_files_dirs = order_arranging(rest_list, mem_list, math_list, motor_list)


validation_file_dir = root + "Data/validation"
all_validate_files = [f for f in listdir(validation_file_dir) if isfile(join(validation_file_dir, f))]
validate_files_dirs = []
for i in range(len(all_validate_files)):
    validate_files_dirs.append(validation_file_dir+'/'+all_validate_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(validate_files_dirs)
validate_files_dirs = order_arranging(rest_list, mem_list, math_list, motor_list)


test_file_dir = root + "Data/test"
all_test_files = [f for f in listdir(test_file_dir) if isfile(join(test_file_dir, f))]
test_files_dirs = []
for i in range(len(all_test_files)):
    test_files_dirs.append(test_file_dir+'/'+all_test_files[i])
rest_list, mem_list, math_list, motor_list = separate_list(test_files_dirs)
test_files_dirs = order_arranging(rest_list, mem_list, math_list, motor_list)