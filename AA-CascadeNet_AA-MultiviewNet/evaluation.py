import os
import numpy as np
from Training import get_multiview_model_gat
import data_utils as utils
import gc
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


folder_test = "./preprocessed/test/"
list_subjects_test = ['204521','212318','162935','601127','725751','735148']
# list_subjects_test = ['204521']
model_string = "C:/Users/lucas/Documenten/Actief/Universiteit/Master/Thesis/backup/multiviewGAT/AA-CascadeNet_AA-MultiviewNet/Experiments/Multiview/Experiment184/15_model184_tf"
batch_size = 16

depth = 10
gamma = 0.1
gat_heads = 3
encoder_heads = 8
threshold = 0.9999
mode = "topk"
k = 3


def load_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    print("loading model")
    saved_model = tf.saved_model.load(model_string)
    print("model loaded")
    model,model_object = get_multiview_model_gat(depth, gamma, gat_heads, encoder_heads, threshold, mode, k)
    look_up = look_up_table()
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

    # saved_model = tf.saved_model.load("C:/Users/lucas/Documenten/Actief/Universiteit/Master/Thesis/aa/AA-CascadeNet_AA-MultiviewNet/Experiments/Multiview/Experiment995_SSL/10_model995_tf")
    return model

def predict(model):
    for subject in list_subjects_test:
        print("\nTesting on subject", subject)
        print("loading files")
        print(model.summary())
        save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
        save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))

        X_test = np.load(save_path_X, allow_pickle=True).item()
        Y_test = np.load(save_path_Y, allow_pickle=True)
        print("files loaded")
        # predictions = model(X_test, training=False)
        predictions = model.predict(X_test, batch_size = batch_size, verbose=1)
        # predicted.argmax(axis=1)
        predictions = predictions.argmax(axis=1)
        Y_test = Y_test.argmax(axis=1)

        print("predicted")
        acc = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions, average='macro')
        precision = precision_score(Y_test, predictions, average='macro')
        recall = recall_score(Y_test, predictions, average='macro')
        confusion_matrix_func(predictions, Y_test, subject)
        print("Results:")
        print(f"Accuracy = [{acc}] - F1 = [{f1}] - Precision = [{precision}] - Recall = [{recall}]")
        X_test = None
        Y_test = None
        gc.collect()


def attention(model):
    
    subject =  list_subjects_test[0]
    print("\nTesting on subject", subject)
    subject_files_test = []
    for item in utils.test_files_dirs:
        if subject in item:
            subject_files_test.append(item)

    save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
    save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))

    X_test = np.load(save_path_X, allow_pickle=True).item()
    Y_test = np.load(save_path_Y, allow_pickle=True)


    for w in range(10):
        X_test['input{}'.format(w+1)] = X_test['input{}'.format(w+1)][0:160]
    print(X_test['input1'].shape)
    
    attention = model.predict(X_test, batch_size = batch_size)

    l = 0

    for layer in attention:
        l += 1
        h=0
        for head in layer:
            h+= 1
            np.save('./evaluation/attention/GAT_{}_h{}'.format(l, h), head)



def evaluate(model):
    accuracies_temp = []
    
    for subject in list_subjects_test:
        print("\nTesting on subject", subject)
        subject_files_test = []
        for item in utils.test_files_dirs:
            if subject in item:
                subject_files_test.append(item)

        save_path_X = os.path.join(folder_test, "X-{}.npy".format(subject))
        save_path_Y = os.path.join(folder_test, "Y-{}.npy".format(subject))

        X_test = np.load(save_path_X, allow_pickle=True).item()
        Y_test = np.load(save_path_Y, allow_pickle=True)

        predictions = model.predict(X_test, batch_size = batch_size)
        confusion_matrix_func(predictions, Y_test)

        result = model.evaluate(X_test, Y_test, batch_size = batch_size,verbose=1)
        X_test = None
        Y_test = None
        gc.collect()
        accuracies_temp.append(result[1]* 100)
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

def look_up_table():
    return {
        'graph_attention/kernel_0:0':'gat_1/gat_1_kernel_0:0',
        'graph_attention/bias_0:0':'gat_1/gat_1_bias_0:0',
        'graph_attention/attn_kernel_self_0:0':'gat_1/gat_1_attn_kernel_self_0:0',
        'graph_attention/attn_kernel_neigh_0:0':'gat_1/gat_1_attn_kernel_neigh_0:0',
        'graph_attention/kernel_1:0':'gat_1/gat_1_kernel_1:0',
        'graph_attention/bias_1:0':'gat_1/gat_1_bias_1:0',
        'graph_attention/attn_kernel_self_1:0':'gat_1/gat_1_attn_kernel_self_1:0',
        'graph_attention/attn_kernel_neigh_1:0':'gat_1/gat_1_attn_kernel_neigh_1:0',
        'graph_attention/kernel_2:0':'gat_1/gat_1_kernel_2:0',
        'graph_attention/bias_2:0':'gat_1/gat_1_bias_2:0',
        'graph_attention/attn_kernel_self_2:0':'gat_1/gat_1_attn_kernel_self_2:0',
        'graph_attention/attn_kernel_neigh_2:0':'gat_1/gat_1_attn_kernel_neigh_2:0',
        'graph_attention/ig_delta:0':'gat_1/gat_1_ig_delta:0',
        'graph_attention/ig_non_exist_edge:0':'gat_1/gat_1_ig_non_exist_edge:0',
        'graph_attention_1/kernel_0:0':'gat_2/gat_2_kernel_0:0',
        'graph_attention_1/bias_0:0':'gat_2/gat_2_bias_0:0',
        'graph_attention_1/attn_kernel_self_0:0':'gat_2/gat_2_attn_kernel_self_0:0',
        'graph_attention_1/attn_kernel_neigh_0:0':'gat_2/gat_2_attn_kernel_neigh_0:0',
        'graph_attention_1/kernel_1:0':'gat_2/gat_2_kernel_1:0',
        'graph_attention_1/bias_1:0':'gat_2/gat_2_bias_1:0',
        'graph_attention_1/attn_kernel_self_1:0':'gat_2/gat_2_attn_kernel_self_1:0',
        'graph_attention_1/attn_kernel_neigh_1:0':'gat_2/gat_2_attn_kernel_neigh_1:0',
        'graph_attention_1/kernel_2:0':'gat_2/gat_2_kernel_2:0',
        'graph_attention_1/bias_2:0':'gat_2/gat_2_bias_2:0',
        'graph_attention_1/attn_kernel_self_2:0':'gat_2/gat_2_attn_kernel_self_2:0',
        'graph_attention_1/attn_kernel_neigh_2:0':'gat_2/gat_2_attn_kernel_neigh_2:0',
        'graph_attention_1/ig_delta:0':'gat_2/gat_2_ig_delta:0',
        'graph_attention_1/ig_non_exist_edge:0':'gat_2/gat_2_ig_non_exist_edge:0',
        'graph_attention_2/kernel_0:0':'gat_3/gat_3_kernel_0:0',
        'graph_attention_2/bias_0:0':'gat_3/gat_3_bias_0:0',
        'graph_attention_2/attn_kernel_self_0:0':'gat_3/gat_3_attn_kernel_self_0:0',
        'graph_attention_2/attn_kernel_neigh_0:0':'gat_3/gat_3_attn_kernel_neigh_0:0',
        'graph_attention_2/kernel_1:0':'gat_3/gat_3_kernel_1:0',
        'graph_attention_2/bias_1:0':'gat_3/gat_3_bias_1:0',
        'graph_attention_2/attn_kernel_self_1:0':'gat_3/gat_3_attn_kernel_self_1:0',
        'graph_attention_2/attn_kernel_neigh_1:0':'gat_3/gat_3_attn_kernel_neigh_1:0',
        'graph_attention_2/kernel_2:0':'gat_3/gat_3_kernel_2:0',
        'graph_attention_2/bias_2:0':'gat_3/gat_3_bias_2:0',
        'graph_attention_2/attn_kernel_self_2:0':'gat_3/gat_3_attn_kernel_self_2:0',
        'graph_attention_2/attn_kernel_neigh_2:0':'gat_3/gat_3_attn_kernel_neigh_2:0',
        'graph_attention_2/ig_delta:0':'gat_3/gat_3_ig_delta:0',
        'graph_attention_2/ig_non_exist_edge:0':'gat_3/gat_3_ig_non_exist_edge:0',
        'graph_attention_3/kernel_0:0':'gat_4/gat_4_kernel_0:0',
        'graph_attention_3/bias_0:0':'gat_4/gat_4_bias_0:0',
        'graph_attention_3/attn_kernel_self_0:0':'gat_4/gat_4_attn_kernel_self_0:0',
        'graph_attention_3/attn_kernel_neigh_0:0':'gat_4/gat_4_attn_kernel_neigh_0:0',
        'graph_attention_3/kernel_1:0':'gat_4/gat_4_kernel_1:0',
        'graph_attention_3/bias_1:0':'gat_4/gat_4_bias_1:0',
        'graph_attention_3/attn_kernel_self_1:0':'gat_4/gat_4_attn_kernel_self_1:0',
        'graph_attention_3/attn_kernel_neigh_1:0':'gat_4/gat_4_attn_kernel_neigh_1:0',
        'graph_attention_3/kernel_2:0':'gat_4/gat_4_kernel_2:0',
        'graph_attention_3/bias_2:0':'gat_4/gat_4_bias_2:0',
        'graph_attention_3/attn_kernel_self_2:0':'gat_4/gat_4_attn_kernel_self_2:0',
        'graph_attention_3/attn_kernel_neigh_2:0':'gat_4/gat_4_attn_kernel_neigh_2:0',
        'graph_attention_3/ig_delta:0':'gat_4/gat_4_ig_delta:0',
        'graph_attention_3/ig_non_exist_edge:0':'gat_4/gat_4_ig_non_exist_edge:0',
        'graph_attention_4/kernel_0:0':'gat_5/gat_5_kernel_0:0',
        'graph_attention_4/bias_0:0':'gat_5/gat_5_bias_0:0',
        'graph_attention_4/attn_kernel_self_0:0':'gat_5/gat_5_attn_kernel_self_0:0',
        'graph_attention_4/attn_kernel_neigh_0:0':'gat_5/gat_5_attn_kernel_neigh_0:0',
        'graph_attention_4/kernel_1:0':'gat_5/gat_5_kernel_1:0',
        'graph_attention_4/bias_1:0':'gat_5/gat_5_bias_1:0',
        'graph_attention_4/attn_kernel_self_1:0':'gat_5/gat_5_attn_kernel_self_1:0',
        'graph_attention_4/attn_kernel_neigh_1:0':'gat_5/gat_5_attn_kernel_neigh_1:0',
        'graph_attention_4/kernel_2:0':'gat_5/gat_5_kernel_2:0',
        'graph_attention_4/bias_2:0':'gat_5/gat_5_bias_2:0',
        'graph_attention_4/attn_kernel_self_2:0':'gat_5/gat_5_attn_kernel_self_2:0',
        'graph_attention_4/attn_kernel_neigh_2:0':'gat_5/gat_5_attn_kernel_neigh_2:0',
        'graph_attention_4/ig_delta:0':'gat_5/gat_5_ig_delta:0',
        'graph_attention_4/ig_non_exist_edge:0':'gat_5/gat_5_ig_non_exist_edge:0',
        'graph_attention_5/kernel_0:0':'gat_6/gat_6_kernel_0:0',
        'graph_attention_5/bias_0:0':'gat_6/gat_6_bias_0:0',
        'graph_attention_5/attn_kernel_self_0:0':'gat_6/gat_6_attn_kernel_self_0:0',
        'graph_attention_5/attn_kernel_neigh_0:0':'gat_6/gat_6_attn_kernel_neigh_0:0',
        'graph_attention_5/kernel_1:0':'gat_6/gat_6_kernel_1:0',
        'graph_attention_5/bias_1:0':'gat_6/gat_6_bias_1:0',
        'graph_attention_5/attn_kernel_self_1:0':'gat_6/gat_6_attn_kernel_self_1:0',
        'graph_attention_5/attn_kernel_neigh_1:0':'gat_6/gat_6_attn_kernel_neigh_1:0',
        'graph_attention_5/kernel_2:0':'gat_6/gat_6_kernel_2:0',
        'graph_attention_5/bias_2:0':'gat_6/gat_6_bias_2:0',
        'graph_attention_5/attn_kernel_self_2:0':'gat_6/gat_6_attn_kernel_self_2:0',
        'graph_attention_5/attn_kernel_neigh_2:0':'gat_6/gat_6_attn_kernel_neigh_2:0',
        'graph_attention_5/ig_delta:0':'gat_6/gat_6_ig_delta:0',
        'graph_attention_5/ig_non_exist_edge:0':'gat_6/gat_6_ig_non_exist_edge:0',
        'graph_attention_6/kernel_0:0':'gat_7/gat_7_kernel_0:0',
        'graph_attention_6/bias_0:0':'gat_7/gat_7_bias_0:0',
        'graph_attention_6/attn_kernel_self_0:0':'gat_7/gat_7_attn_kernel_self_0:0',
        'graph_attention_6/attn_kernel_neigh_0:0':'gat_7/gat_7_attn_kernel_neigh_0:0',
        'graph_attention_6/kernel_1:0':'gat_7/gat_7_kernel_1:0',
        'graph_attention_6/bias_1:0':'gat_7/gat_7_bias_1:0',
        'graph_attention_6/attn_kernel_self_1:0':'gat_7/gat_7_attn_kernel_self_1:0',
        'graph_attention_6/attn_kernel_neigh_1:0':'gat_7/gat_7_attn_kernel_neigh_1:0',
        'graph_attention_6/kernel_2:0':'gat_7/gat_7_kernel_2:0',
        'graph_attention_6/bias_2:0':'gat_7/gat_7_bias_2:0',
        'graph_attention_6/attn_kernel_self_2:0':'gat_7/gat_7_attn_kernel_self_2:0',
        'graph_attention_6/attn_kernel_neigh_2:0':'gat_7/gat_7_attn_kernel_neigh_2:0',
        'graph_attention_6/ig_delta:0':'gat_7/gat_7_ig_delta:0',
        'graph_attention_6/ig_non_exist_edge:0':'gat_7/gat_7_ig_non_exist_edge:0',
        'graph_attention_7/kernel_0:0':'gat_8/gat_8_kernel_0:0',
        'graph_attention_7/bias_0:0':'gat_8/gat_8_bias_0:0',
        'graph_attention_7/attn_kernel_self_0:0':'gat_8/gat_8_attn_kernel_self_0:0',
        'graph_attention_7/attn_kernel_neigh_0:0':'gat_8/gat_8_attn_kernel_neigh_0:0',
        'graph_attention_7/kernel_1:0':'gat_8/gat_8_kernel_1:0',
        'graph_attention_7/bias_1:0':'gat_8/gat_8_bias_1:0',
        'graph_attention_7/attn_kernel_self_1:0':'gat_8/gat_8_attn_kernel_self_1:0',
        'graph_attention_7/attn_kernel_neigh_1:0':'gat_8/gat_8_attn_kernel_neigh_1:0',
        'graph_attention_7/kernel_2:0':'gat_8/gat_8_kernel_2:0',
        'graph_attention_7/bias_2:0':'gat_8/gat_8_bias_2:0',
        'graph_attention_7/attn_kernel_self_2:0':'gat_8/gat_8_attn_kernel_self_2:0',
        'graph_attention_7/attn_kernel_neigh_2:0':'gat_8/gat_8_attn_kernel_neigh_2:0',
        'graph_attention_7/ig_delta:0':'gat_8/gat_8_ig_delta:0',
        'graph_attention_7/ig_non_exist_edge:0':'gat_8/gat_8_ig_non_exist_edge:0',
        'graph_attention_8/kernel_0:0':'gat_9/gat_9_kernel_0:0',
        'graph_attention_8/bias_0:0':'gat_9/gat_9_bias_0:0',
        'graph_attention_8/attn_kernel_self_0:0':'gat_9/gat_9_attn_kernel_self_0:0',
        'graph_attention_8/attn_kernel_neigh_0:0':'gat_9/gat_9_attn_kernel_neigh_0:0',
        'graph_attention_8/kernel_1:0':'gat_9/gat_9_kernel_1:0',
        'graph_attention_8/bias_1:0':'gat_9/gat_9_bias_1:0',
        'graph_attention_8/attn_kernel_self_1:0':'gat_9/gat_9_attn_kernel_self_1:0',
        'graph_attention_8/attn_kernel_neigh_1:0':'gat_9/gat_9_attn_kernel_neigh_1:0',
        'graph_attention_8/kernel_2:0':'gat_9/gat_9_kernel_2:0',
        'graph_attention_8/bias_2:0':'gat_9/gat_9_bias_2:0',
        'graph_attention_8/attn_kernel_self_2:0':'gat_9/gat_9_attn_kernel_self_2:0',
        'graph_attention_8/attn_kernel_neigh_2:0':'gat_9/gat_9_attn_kernel_neigh_2:0',
        'graph_attention_8/ig_delta:0':'gat_9/gat_9_ig_delta:0',
        'graph_attention_8/ig_non_exist_edge:0':'gat_9/gat_9_ig_non_exist_edge:0',
        'graph_attention_9/kernel_0:0':'gat_10/gat_10_kernel_0:0',
        'graph_attention_9/bias_0:0':'gat_10/gat_10_bias_0:0',
        'graph_attention_9/attn_kernel_self_0:0':'gat_10/gat_10_attn_kernel_self_0:0',
        'graph_attention_9/attn_kernel_neigh_0:0':'gat_10/gat_10_attn_kernel_neigh_0:0',
        'graph_attention_9/kernel_1:0':'gat_10/gat_10_kernel_1:0',
        'graph_attention_9/bias_1:0':'gat_10/gat_10_bias_1:0',
        'graph_attention_9/attn_kernel_self_1:0':'gat_10/gat_10_attn_kernel_self_1:0',
        'graph_attention_9/attn_kernel_neigh_1:0':'gat_10/gat_10_attn_kernel_neigh_1:0',
        'graph_attention_9/kernel_2:0':'gat_10/gat_10_kernel_2:0',
        'graph_attention_9/bias_2:0':'gat_10/gat_10_bias_2:0',
        'graph_attention_9/attn_kernel_self_2:0':'gat_10/gat_10_attn_kernel_self_2:0',
        'graph_attention_9/attn_kernel_neigh_2:0':'gat_10/gat_10_attn_kernel_neigh_2:0',
        'graph_attention_9/ig_delta:0':'gat_10/gat_10_ig_delta:0',
        'graph_attention_9/ig_non_exist_edge:0':'gat_10/gat_10_ig_non_exist_edge:0',
        'transformer_encoder/self_attention_layer/query/kernel:0':'encoder_block/self_attention_layer/query/kernel:0',
        'transformer_encoder/self_attention_layer/query/bias:0':'encoder_block/self_attention_layer/query/bias:0',
        'transformer_encoder/self_attention_layer/key/kernel:0':'encoder_block/self_attention_layer/key/kernel:0',
        'transformer_encoder/self_attention_layer/key/bias:0':'encoder_block/self_attention_layer/key/bias:0',
        'transformer_encoder/self_attention_layer/value/kernel:0':'encoder_block/self_attention_layer/value/kernel:0',
        'transformer_encoder/self_attention_layer/value/bias:0':'encoder_block/self_attention_layer/value/bias:0',
        'transformer_encoder/self_attention_layer/attention_output/kernel:0':'encoder_block/self_attention_layer/attention_output/kernel:0',
        'transformer_encoder/self_attention_layer/attention_output/bias:0':'encoder_block/self_attention_layer/attention_output/bias:0',
        'transformer_encoder/gamma:0':'encoder_block/gamma:0',
        'transformer_encoder/beta:0':'encoder_block/beta:0',
        'transformer_encoder/gamma:0':'encoder_block/gamma:0',
        'transformer_encoder/beta:0':'encoder_block/beta:0',
        'transformer_encoder/kernel:0':'encoder_block/kernel:0',
        'transformer_encoder/bias:0':'encoder_block/bias:0',
        'transformer_encoder/kernel:0':'encoder_block/kernel:0',
        'transformer_encoder/bias:0':'encoder_block/bias:0',
        'dense_10/kernel:0':'encoder_dense/kernel:0',
        'dense_10/bias:0':'encoder_dense/bias:0',
        'dense/kernel:0':'gat_dense_1/kernel:0',
        'dense/bias:0':'gat_dense_1/bias:0',
        'dense_1/kernel:0':'gat_dense_2/kernel:0',
        'dense_1/bias:0':'gat_dense_2/bias:0',
        'dense_2/kernel:0':'gat_dense_3/kernel:0',
        'dense_2/bias:0':'gat_dense_3/bias:0',
        'dense_3/kernel:0':'gat_dense_4/kernel:0',
        'dense_3/bias:0':'gat_dense_4/bias:0',
        'dense_4/kernel:0':'gat_dense_5/kernel:0',
        'dense_4/bias:0':'gat_dense_5/bias:0',
        'dense_5/kernel:0':'gat_dense_6/kernel:0',
        'dense_5/bias:0':'gat_dense_6/bias:0',
        'dense_6/kernel:0':'gat_dense_7/kernel:0',
        'dense_6/bias:0':'gat_dense_7/bias:0',
        'dense_7/kernel:0':'gat_dense_8/kernel:0',
        'dense_7/bias:0':'gat_dense_8/bias:0',
        'dense_8/kernel:0':'gat_dense_9/kernel:0',
        'dense_8/bias:0':'gat_dense_9/bias:0',
        'dense_9/kernel:0':'gat_dense_10/kernel:0',
        'dense_9/bias:0':'gat_dense_10/bias:0',
        'dense_11/kernel:0':'dense/kernel:0',
        'dense_11/bias:0':'dense/bias:0',
    }

model = load_model()
attention(model)
predict(model)

