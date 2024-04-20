import os
import random
import time
import gc
import sys
import numpy as np
import tensorflow
from MultiviewGAT import MultiviewGAT

import experiment_utils as eutils
import data_utils as utils

from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

window_size = 10
dense_activation = "relu"
dense3_activation = "relu"

def get_multiview_model_gat(depth, gamma, gat_heads, encoder_heads, threshold, mode, k):
    multiview_attention_object = MultiviewGAT(window_size,dense_activation,
             dense3_activation, depth, gamma,
             gat_heads, encoder_heads, threshold, mode, k)
    multiview_attention_model = multiview_attention_object.model
    return multiview_attention_model, multiview_attention_object

train_loss_results = []
train_accuracy_results = []

def train(setup, mode, threshold=0.95, k=3, num_epochs=15,depth=10,batch_size=32, gamma=100, preprocess=False, gat_heads=3, encoder_heads=8, lr=0.0001, overlap=0.5):
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    np.random.seed(123)
    tensorflow.random.set_seed(1234)
    os.environ['PYTHONHASHSEED']=str(1234)
    random.seed(1234)

    model_type = "Multiview"
    if setup == 0:#used for quick tests
        subjects = ['105923']
        list_subjects_test = ['212318']
    if setup == 1:
        subjects = ['105923','164636','133019']
        list_subjects_test = ['204521','212318','162935','601127','725751','735148']
    if setup == 2:
        subjects = ['105923','164636','133019','113922','116726','140117','175237','177746','185442','191033','191437','192641']
        list_subjects_test = ['204521','212318','162935','601127','725751','735148']
    
    model,model_object = get_multiview_model_gat(depth, gamma, gat_heads, encoder_heads, threshold, mode, k)

    batch_size = batch_size

    subjects_string = ",".join([subject for subject in subjects])
    comment = "Training with subjects : " + subjects_string
    
    accuracies_temp_train = []
    losses_temp_train= []
    
    accuracies_train = []#per epoch
    losses_train = []#per epoch
    
    accuracies_temp_val = []
    losses_temp_val = []
    
    accuracies_val = []#per epoch
    losses_val = []#per epoch
    start_time = time.time()

    model.compile(optimizer = Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    experiment_number = eutils.on_train_begin(model_object,model_type,setup,batch_size)

    folder_train = "./preprocessed/train/"
    if not os.path.exists(folder_train):
        os.makedirs(folder_train)
    folder_validate = "./preprocessed/validate/"
    if not os.path.exists(folder_validate):
        os.makedirs(folder_validate)
    folder_test = "./preprocessed/test/"
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)

    for epoch in range(num_epochs):
        print("\n"+ 32*"#"+ "[Epoch {}]".format(epoch+1) + 32*"#" + "\n")
        start_epoch = time.time()
        for subject in subjects:
            start_subject_time = time.time()
            print("Training on subject [{}]".format(subject))
            print(" --Loading training data subject [{}]".format(subject), end="\r")
            
            save_path_X = os.path.join(folder_train, "X-{}.npy".format(subject))                
            save_path_Y = os.path.join(folder_train, "Y-{}.npy".format(subject))
            X_train = None
            Y_train = None

            if(not os.path.isfile(save_path_X) or ( preprocess and epoch == 0)):
                subject_files_train = []
                for item in utils.train_files_dirs:
                    if subject in item:
                        subject_files_train.append(item)

                number_workers_training = 1
                number_files_per_worker = len(subject_files_train)//number_workers_training

                X_train, Y_train = utils.multi_processing_multiviewGAT(subject_files_train,number_files_per_worker,number_workers_training,depth)
                np.save(save_path_X, X_train)
                np.save(save_path_Y, Y_train)
            else:
                X_train = np.load(save_path_X, allow_pickle=True).item()
                Y_train = np.load(save_path_Y, allow_pickle=True)

            X_train,Y_train = utils.reshape_input_dictionary(X_train, Y_train, batch_size, depth)
            print(" ++Training data loaded [{}]\t\t".format(subject))
            print(" --Loading validation data subject [{}]".format(subject), end='\r')

            save_path_X = os.path.join(folder_validate, "X-{}.npy".format(subject))
            save_path_Y = os.path.join(folder_validate, "Y-{}.npy".format(subject))
            X_validate = None
            Y_validate = None

            if(not os.path.isfile(save_path_X) or ( preprocess and epoch == 0)):
                subject_files_val = []
                for item in utils.validate_files_dirs:
                    if subject in item:
                        subject_files_val.append(item)

                number_workers_validation = 8
                number_files_per_worker = len(subject_files_val)//number_workers_validation
                X_validate, Y_validate = utils.multi_processing_multiviewGAT(subject_files_val,number_files_per_worker,number_workers_validation,depth)       
                np.save(save_path_X, X_validate)
                np.save(save_path_Y, Y_validate)
            else:
                X_validate = np.load(save_path_X, allow_pickle=True).item()
                Y_validate = np.load(save_path_Y, allow_pickle=True)

            X_validate, Y_validate = utils.reshape_input_dictionary(X_validate, Y_validate, batch_size,depth)

            print(" ++Validation data loaded [{}]\t\t".format(subject))
            history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, 
                                    verbose = 1, validation_data=(X_validate, Y_validate), 
                                    callbacks=None) 
            eutils.model_save(experiment_number,model,model_type,epoch+1)
            print("Model and model's weights saved, Epoch : {}, subject : {}".format(epoch+1,subject) )
            print("Timespan subject training is : {:.2f}s".format(time.time() - start_subject_time))
            print("")
            history_dict = history.history
            accuracies_temp_train.append(history_dict['accuracy'][0] * 100)
            losses_temp_train.append(history_dict['loss'][0])
            accuracies_temp_val.append(history_dict['val_accuracy'][0] * 100)
            losses_temp_val.append(history_dict['val_loss'][0])
            
            #Freeing memory
            X_train = None
            Y_train = None
            X_validate = None
            Y_validate = None
            gc.collect()
        
        ## Training Information ##
        print("\n--Epoch [{}] summary".format(epoch+1))
        average_loss_epoch_train = sum(losses_temp_train)/len(losses_temp_train)
        print("Epoch Training Loss : {:.3f}".format(average_loss_epoch_train))
        losses_train.append(average_loss_epoch_train)
        losses_temp_train = []

        average_accuracy_epoch_train = sum(accuracies_temp_train)/len(accuracies_temp_train)
        print("Epoch Training Accuracy: {:.3%}".format(average_accuracy_epoch_train / 100))
        accuracies_train.append(average_accuracy_epoch_train)
        accuracies_temp_train = []

        ## Validation Information ##
        average_loss_epoch_validate = sum(losses_temp_val)/len(losses_temp_val)
        print("Epoch Validation Loss : {:.3f}".format(average_loss_epoch_validate))
        losses_val.append(average_loss_epoch_validate)
        losses_temp_val = []

        average_accuracy_epoch_validate = sum(accuracies_temp_val)/len(accuracies_temp_val)
        print("Epoch Validation Accuracy: {:.3%}".format(average_accuracy_epoch_validate / 100))
        accuracies_val.append(average_accuracy_epoch_validate)
        accuracies_temp_val = []

        print("==Epoch training duration {:2f} seconds ==".format(time.time() - start_epoch))
        eutils.on_epoch_end(epoch, average_accuracy_epoch_train, average_loss_epoch_train, \
                        average_accuracy_epoch_validate, average_loss_epoch_validate, experiment_number, model,model_type)

        if (epoch+1) % 2 == 0 or (epoch+1) == num_epochs:
            accuracies_temp = []
            for subject in list_subjects_test:
                start_testing = time.time()
                print("\nTesting on subject", subject)
                subject_files_test = []
                for item in utils.test_files_dirs:
                    if subject in item:
                        subject_files_test.append(item)
                            
                number_workers_testing = 10
                number_files_per_worker = len(subject_files_test)//number_workers_testing

                save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
                save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))
                if(not os.path.isfile(save_path_X) or ( preprocess and epoch+1 == 2)):
                    X_test, Y_test = utils.multi_processing_multiviewGAT(subject_files_test,number_files_per_worker,number_workers_testing,depth)
                    np.save(save_path_X, X_test)
                    np.save(save_path_Y, Y_test)
                else:
                    X_test = np.load(save_path_X, allow_pickle=True).item()
                    Y_test = np.load(save_path_Y, allow_pickle=True)

                for w in range(10):
                    X_test["input"+str(w+1)][:,124] = [0]*10

                result = model.evaluate(X_test, Y_test, batch_size = 8,verbose=1)
                X_test = None
                Y_test = None
                gc.collect()
                accuracies_temp.append(result[1]* 100)
                print("Recording the testing accuracy of [{}] in a file".format(subject))
                eutils.append_individual_test(experiment_number,epoch+1,subject,result[1] * 100,model_type)
                print("Timespan of testing is : {}".format(time.time() - start_testing))
            avg = sum(accuracies_temp)/len(accuracies_temp)
            print("\n--Test summary epoch [{}]".format(epoch + 1))
            print("Average testing accuracy : {0:.2f}".format(avg))
            print("Standard deviation: {}".format(np.std(accuracies_temp)))
            print("Recording the average testing accuracy in a file")
            eutils.append_average_test(experiment_number,epoch+1,avg,model_type)

            X_test = None
            Y_test = None

    time_span = time.time() - start_time
    print()
    print()
    print("Training took {:.2f} seconds".format(time_span))
    eutils.on_train_end(experiment_number,model_type)
    eutils.save_training_time(experiment_number, time_span,model_type)
    eutils.write_comment(experiment_number,comment,model_type,setup)

import argparse

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--setup', type=int, help="Please select a number between \
                        0 and 2 to choose the setup of the training", default=0)  
    parser.add_argument('-adj','--adjacency',type=str,help="Please choose the type of adjacency",\
                        choices=['none','threshold','topk'],default = "topk")
    parser.add_argument('-t','--threshold',type=float,help="Please choose a threshold value\
                        , by default 0.95", default=0.95)    
    parser.add_argument('-k',type=float,help="Please choose a K value\
                        , by default 3", default=3)

    args = parser.parse_args()
        
    if args.setup < 0 or args.setup > 3:
        print("Invalid setup number, exiting ...")
        sys.exit()

    train(args.setup, args.adjacency, args.threshold, args.k, preprocess=False)

