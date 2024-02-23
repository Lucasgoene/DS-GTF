import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import boto3
import shutil
import numpy as np
from os.path import isdir,isfile,join,exists
from os import mkdir,makedirs,getcwd,listdir
import mne
import reading_raw
from sklearn.metrics.pairwise import rbf_kernel, chi2_kernel
import networkx as nx 
from data_utils import get_dataset_name, get_raw_coordinates, order_arranging, separate_list

folder = "./plot_data/"

def get_raw_data():
    subject = '140117'
    type_state='rest'
    hcp_path = getcwd()

    raw = reading_raw.read_raw(subject=subject, hcp_path=hcp_path, run_index=0, data_type=type_state)
    print(raw.n_times)
    print(raw.times[2023])
    print(raw.info['sfreq'])
    # coords = [0] * 248
    # for row in raw.info['chs']:
    #     name = row['ch_name']
    #     if([*name][0] =='A'):
    #         index = [*name][1:]
    #         coords[int("".join(index)) - 1] = (row['loc'][0:3])

    # return coords

def plot_data(matrix_list):
    for i in range(4):
        matrix = matrix_list[i]
        matrix[236] = np.average(matrix[:,10000])
        matrix[224] = np.average(matrix[:,10000])
    
    # print(len(matrix[0]))
    # print(1 + 'a')

    # matrix = matrix_list[3]
    # matrix_alt = matrix[0:8]
    # f, ax = plt.subplots(8, sharex=True, sharey=True, figsize=(15,8))

    # ax[0].set_title('Sensor measurements per channel (limited to first 8 channels)')

    # step = 1
    # stop = 1001
    # label_num = 10
    # for i in range(len(matrix_alt)):
    #     ax[i].plot(matrix_alt[i][1:stop:step])
    #     ax[i].set_yticks([])
    #     # ax[i].set_xticks(np.arange(0,len(matrix[i][1:stop+1:step]),(len(matrix[i][1:stop+1:step])) // label_num), np.arange(0,len(matrix[i][1:stop+1:step]),(len(matrix[i][1:stop+1:step])) // label_num)*step)
    #     ax[i].set_xticks([])
    #     ax[i].margins(x=0)

    # f.subplots_adjust(hspace=0)
    # plt.savefig("channel_data.pdf")
    # plt.show()

    # plt.clf()
    # plt.figure(figsize = (15,4))
    # plt.imshow(matrix_alt[:,1:101:step], cmap='plasma')
    # plt.yticks([])
    # plt.xticks([])
    # # plt.savefig("segment1.pdf")

    # plt.clf()
    # plt.imshow(matrix_alt[:,51:151:step], cmap='plasma')
    # plt.yticks([])
    # plt.xticks([])
    # # plt.savefig("segment2.pdf")

    # plt.clf()

    # f, ax = plt.subplots(1, 10, sharey=True, figsize=(15,8))

    # step = 1
    # stop = 101
    # label_num = 10
    # for i in range(10):
    #     ax[i].imshow(matrix_alt[:,(i*10)+1:1+((i+1)*10):], cmap='plasma')
    #     ax[i].set_yticks([])
    #     # ax[i].set_xticks(np.arange(0,len(matrix[i][1:stop+1:step]),(len(matrix[i][1:stop+1:step])) // label_num), np.arange(0,len(matrix[i][1:stop+1:step]),(len(matrix[i][1:stop+1:step])) // label_num)*step)
    #     ax[i].set_xticks([])
    #     ax[i].margins(x=0)


    # plt.savefig("window1.pdf")
    # plt.show()

    titles = ["Resting", "Memory", "Story & Math", "Motor"]

    f, ax = plt.subplots(2,2, sharex=False, sharey=True, figsize=(20,10))
    minmin = np.min([np.min(matrix_list[0][:,1::75]), np.min(matrix_list[1][:,1::75]), np.min(matrix_list[2][:,1::75]), np.min(matrix_list[3][:,1::75])])
    maxmax = np.max([np.max(matrix_list[0][:,1::75]), np.max(matrix_list[1][:,1::75]), np.max(matrix_list[2][:,1::75]), np.max(matrix_list[3][:,1::75])])

    for i in range(2):
        for j in range(2):
            im = ax[i][j].imshow(matrix_list[i*2+j][:,1::75], vmin = minmin, vmax= maxmax, cmap='plasma', aspect='auto')
            ax[i][j].set_yticks([])
            # print(np.arange(0,len(matrix[i][1::75])+1,len(matrix[i][1::75]) // 4)*75)
            ax[i][j].set_xticks(np.arange(0,len(matrix[i][1::75]) + 1,len(matrix[i][1::75]) // 4), np.arange(0,len(matrix[i][1::75])+1,len(matrix[i][1::75]) // 4)*75, fontsize=24)
            # ax[i][j].set_xticklabels(np.arange(0,len(matrix[i][1::75])+1,len(matrix[i][1::75]) // 4)*75)
            ax[i][j].set_title(titles[i*2+j], size=32)
            ax[i][j].set_xlabel("timesteps", size=28)
            ax[i][j].set_ylabel("channels", size=28)
    
    f.tight_layout()
    f.subplots_adjust(right=0.83, hspace = 0.5)
    cbar_ax = f.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = f.colorbar(im, cax=cbar_ax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(24)
    for tt in cbar.ax.get_xlabel():
        tt.set_fontsize(24)
    for tt in cbar.ax.get_xticks():
        tt.set_fontsize(24)
    # plt.font_scale(1.5)
    # plt.colorbar()
    plt.savefig("matrix_all_classes.pdf")
    plt.show()

    # # plt.figure(figsize = (15,4))
    # for i in range(4):
    #     plt.figure(figsize = (15,4))
    #     plt.imshow(matrix_list[i][:,1::75], cmap='plasma', aspect='auto', )
    #     plt.yticks([])
    #     plt.xticks(np.arange(0,len(matrix[i][1::75]) + 1,len(matrix[i][1::75]) // 4), np.arange(0,len(matrix[i][1::75])+1,len(matrix[i][1::75]) // 4)*75)
    #     plt.title(titles[i])
    #     plt.xlabel("timesteps")
    #     plt.ylabel("channels")
    #     plt.colorbar()
    #     plt.savefig("matrix_{}.pdf".format(titles[i]))
    #     plt.show()

        # plt.clf()
        

    # matrix = matrix[:,1::750]
    # plt.imshow(matrix, cmap='plasma')
    # plt.yticks([])

    # plt.savefig("matrix_all_classes.pdf")
    # plt.show()

def plot_graph(assignment):
    
    coords = get_raw_coordinates()
    node_xyz = np.array(coords)
    edge_xyz = np.array([])
    edge_att = np.array([])

    kernel = np.zeros((248,248))

    centroids = np.zeros(8)

    for i in range(8):
        index = np.where(assignment == i)
        ix = len(index[0]) // 2
        centroids[i] = index[0][ix]
    print(centroids)

    # for i in range(8):
    #     print(i)
    #     ind = np.where(assignment == i)
    #     print(np.array(ind))
    #     print(ind)
    for x in range(248):
        ass = assignment[x]
        y = int(centroids[int(ass)])
        # for y in range(248):
            # if(assignment[x] == assignment[y]):
        if len(edge_xyz) == 0:
            edge_xyz = np.array([(node_xyz[x],node_xyz[y])])
        else:
            arr = np.array([(node_xyz[x],node_xyz[y])])
            edge_xyz = np.concatenate((edge_xyz, arr))
                # kernel[i][j] = 1

    for a in range(len(centroids)):
        for b in range(len(centroids)):
            arr = np.array([(node_xyz[int(centroids[a])],node_xyz[int(centroids[b])])])
            edge_xyz = np.concatenate((edge_xyz, arr))


    fig = plt.figure()
    # assignment = np.load("./k2-assignment.npy")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*node_xyz.T, s=30, ec="w", c=assignment)
    i = 0
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="gray")
        i += 1
    fig.tight_layout()
    plt.show()

def plot_kernel(attention, gamma = 100, threshold = 0.95):
    # kernel = np.load('./graph.npy')
    coords = get_raw_coordinates()
    print(f"gamma is {gamma}")
    print(f"threshold is {threshold}")
    #RBF 
    kernel = rbf_kernel(coords, gamma = gamma)
    # plt.imshow(kernel)
    # plt.show()
    
    G = nx.DiGraph() 
    node_xyz = np.array(coords[,:2])
    edge_xyz = np.array([])
    edge_att = np.array([])
            
    for channel in kernel:
        indices = np.argpartition(channel,-3)[-3:]
        mask = np.ones(248, dtype=bool)
        mask[indices] = False
        channel[mask] = 0.0
        channel[~mask] = 1.0

    for i in range(248):
        for j in range(248):
            if(kernel[i][j] == 1.0):
                if (i != j):
                    if len(edge_xyz) == 0:
                        edge_xyz = np.array([(node_xyz[i],node_xyz[j])])
                        edge_att = np.array([attention[0][i][j]])
                    else:
                        arr = np.array([(node_xyz[i],node_xyz[j])])
                        arr_att = np.array([attention[0][i][j]])
                        edge_xyz = np.concatenate((edge_xyz, arr))
                        edge_att = np.concatenate((edge_att, arr_att))
            # if(abs(coords[i][2] - coords[j][2]) < 0.02 and (abs(coords[i][0] - coords[j][0]) < 0.01) and (coords[i][2] > 0.04 and coords[i][2] < 0.06 and coords[j][2] > 0.04 and coords[j][2] < 0.06 )):
            #     kernel[i][j] = 1.0
            #     if (i != j):
            #         if len(edge_xyz) == 0:
            #             edge_xyz = np.array([(node_xyz[i],node_xyz[j])])
            #         else:
            #             arr = np.array([(node_xyz[i],node_xyz[j])])
            #             edge_xyz = np.concatenate((edge_xyz, arr))

            # if(abs(coords[i][2] - coords[j][2]) < 0.02 and (abs(coords[i][0] - coords[j][0]) < 0.01) and (coords[i][2] > 0.06 and coords[i][2] < 0.08 and coords[j][2] > 0.06 and coords[j][2] < 0.08 )):
            #     kernel[i][j] = 1.0
            #     if (i != j):
            #         if len(edge_xyz) == 0:
            #             edge_xyz = np.array([(node_xyz[i],node_xyz[j])])
            #         else:
            #             arr = np.array([(node_xyz[i],node_xyz[j])])
            #             edge_xyz = np.concatenate((edge_xyz, arr))


    # plt.imshow(kernel)
    # plt.show()

    fig = plt.figure()
    assignment = np.load("./k2-assignment.npy")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*node_xyz.T, s=30, ec="w", c=assignment)
    i = 0
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="gray", linewidth =edge_att[i]*15)
        i += 1
    fig.tight_layout()
    plt.show()
    # plt.show() 
    
    # G = nx.DiGraph() 
    # edge_xyz = np.array([])
    # kernel = rbf_kernel(coords, gamma = gamma)
    # for channel in kernel:
    #     # indices = np.argpartition(kernel[i],-k)[-k:]
    #     indices = np.argpartition(channel,-3)[-3:]
    #     mask = np.ones(248, dtype=bool)
    #     mask[indices] = False
    #     channel[mask] = 0.0
    #     channel[~mask] = 1.0

    # for i in range(248):
    #     for j in range(248):
    #         if kernel[i][j] == 1:
    #             if (i != j):
    #                 if len(edge_xyz) == 0:
    #                     edge_xyz = np.array([(node_xyz[i],node_xyz[j])])
    #                 else:
    #                     arr = np.array([(node_xyz[i],node_xyz[j])])
    #                     edge_xyz = np.concatenate((edge_xyz, arr))

    # # plt.imshow(kernel)
    # # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(*node_xyz.T, s=100, ec="w")
    # for vizedge in edge_xyz:
    #     ax.plot(*vizedge.T, color="tab:gray")
    # fig.tight_layout()
    # plt.show() 

rest = np.load(folder + "rest_matrix.npy")
memory = np.load(folder + "memory_matrix.npy")
motor = np.load(folder + "motor_matrix.npy")
math = np.load(folder + "math_matrix.npy")

attention_old = np.load('./evaluation/attention/GAT_1_h1.npy')

assignment = np.load("./k2-assignment.npy")
# plot_graph(assignment)

# get_raw_data()
# plot_data([rest,memory, math, motor])
for j in range(10):
    for i in range(3):
        print("GAT: {} head: {}".format(j+1, i+1))
        attention_new = np.load('./evaluation/attention/GAT_{}_h{}.npy'.format(j+1, i+1))
        for a in range(len(attention_new[0])):
            for b in range(len(attention_new[0][a])):
                if (abs(attention_new[0][a][b] - attention_old[0][a][b]) > 0.01 and (j > 0 or i > 1)) :
                    print(attention_new[0][a][b])
                    print(attention_old[0][a][b])
                    print("==============")
        plot_kernel(attention_new)
        attention_old = attention_new
