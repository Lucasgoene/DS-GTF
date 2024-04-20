import argparse
import os
import numpy as np
from training import get_multiview_model_gat
import data_utils as utils
import gc
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.optimizers import Adam
from numba import cuda 

folder_test = "./preprocessed/test/"
list_subjects_test = ['204521','212318','162935','601127','725751','735148']

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="top_k", help ="Please define a model")
args = parser.parse_args()


model_string = "best_models/{}/model".format(args.model)
batch_size = 16

# From paper
depth = 10
gamma = 0.1
gat_heads = 3
encoder_heads = 8
threshold = 0.95
mode = "topk"
k = 3
lr = 0.0001


def load_model():
    print("loading model")
    saved_model = tf.saved_model.load(model_string)
    print("model loaded")
    model,_ = get_multiview_model_gat(depth, gamma, gat_heads, encoder_heads, threshold, mode, k)
    variables = saved_model.variables

    gamma_count = 0
    beta_count = 0
    kernel_count = 0
    bias_count = 0
    for layer in model.layers:
        for weight in layer.weights:
            #Find saved weights
            saved_layers = [v for v in variables if v.name == weight.name]
            #Check if not null
            if(saved_layers != []):
                print("Initializing layer {} with weights \t\t\t\t\t\t\t\t\t".format(saved_layers[0].name), end='\r')
                if("encoder_block/gamma:0" in saved_layers[0].name):
                    weight.assign(saved_layers[gamma_count])
                    gamma_count += 1
                    continue
                if("encoder_block/beta:0" in saved_layers[0].name):
                    weight.assign(saved_layers[beta_count])
                    beta_count += 1
                    continue
                if("encoder_block/kernel:0" in saved_layers[0].name):
                    weight.assign(saved_layers[kernel_count])
                    kernel_count += 1
                    continue
                if("encoder_block/bias:0" in saved_layers[0].name):
                    weight.assign(saved_layers[bias_count])
                    bias_count += 1
                    continue
                weight.assign(saved_layers[0])
    print(model.summary())
    return model

def predict(model): 
    for subject in list_subjects_test:
        print("\nTesting on subject", subject)
        print("Loading files")
        X_test = None
        Y_test = None
        gc.collect()
        subject_files_test = []
        for item in utils.test_files_dirs:
            if subject in item:
                subject_files_test.append(item)
        number_workers_testing = 10
        number_files_per_worker = len(subject_files_test)//number_workers_testing

        save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
        save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))
        if(not os.path.isfile(save_path_X)):
            X_test, Y_test = utils.multi_processing_multiviewGAT(subject_files_test,number_files_per_worker,number_workers_testing,depth)
            np.save(save_path_X, X_test)
            np.save(save_path_Y, Y_test)
        else:
            X_test = np.load(save_path_X, allow_pickle=True).item()
            Y_test = np.load(save_path_Y, allow_pickle=True)

        print("Files loaded")
        predictions = model.predict(X_test, batch_size = batch_size, verbose=1)
        predictions = predictions.argmax(axis=1)
        Y_test = Y_test.argmax(axis=1)

        acc = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions, average='macro')
        precision = precision_score(Y_test, predictions, average='macro')
        recall = recall_score(Y_test, predictions, average='macro')
        # confusion_matrix_func(predictions, Y_test, subject)
        print("Results:")
        print(f"Accuracy = [{acc}] - F1 = [{f1}] - Precision = [{precision}] - Recall = [{recall}]")
        X_test = None
        Y_test = None
        gc.collect()

def evaluate(model):
    accuracies_temp = []
    for subject in list_subjects_test:
        print("\nTesting on subject", subject)

        subject_files_test = []
        for item in utils.test_files_dirs:
            if subject in item:
                subject_files_test.append(item)
        number_workers_testing = 10
        number_files_per_worker = len(subject_files_test)//number_workers_testing

        save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
        save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))
        if(not os.path.isfile(save_path_X)):
            X_test, Y_test = utils.multi_processing_multiviewGAT(subject_files_test,number_files_per_worker,number_workers_testing,depth)
            np.save(save_path_X, X_test)
            np.save(save_path_Y, Y_test)
        else:
            X_test = np.load(save_path_X, allow_pickle=True).item()
            Y_test = np.load(save_path_Y, allow_pickle=True)

        # predictions = model.predict(X_test, batch_size = batch_size)
        # confusion_matrix_func(predictions, Y_test)
        model.compile(optimizer = Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
        result = model.evaluate(X_test, Y_test, batch_size = batch_size,verbose=1)
        X_test = None
        Y_test = None
        accuracies_temp.append(result[1]* 100)
        result = None
        gc.collect()
    avg = sum(accuracies_temp)/len(accuracies_temp)
    print("\n--Test summary")
    print("Average testing accuracy : {0:.2f}".format(avg))
    print("Standard deviation: {}".format(np.std(accuracies_temp)))
    print("Recording the average testing accuracy in a file")

def confusion_matrix_func(y_pred, y_true, subject):
    classes = ["rest", "math", "mem","motor"]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.set(font_scale=2.5)
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output_{}.pdf'.format(subject))


gpus = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
model = load_model()

predict(model)
# evaluate(model)
