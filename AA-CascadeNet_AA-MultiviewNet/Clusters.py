import numpy as np

from data_utils import get_raw_coordinates

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

def k_means(input, cluster_count, max_iter=300):
    input_len = len(input[0])

    channels = np.arange(0,248)
    assignment = np.zeros(248)
    centroids = np.zeros((8,input_len))
    print(centroids)
    prev_centroids = np.zeros((8,input_len))
    iteration = 0


    #Assign centroids
    sample = np.random.choice(248, 8)
    for i in range(cluster_count):
        centroids[i] = input[sample[i]]
    print(sample)
    print("centroids selected")
    print(centroids)
    print(prev_centroids)
    print(np.not_equal(centroids, prev_centroids).any())

    while np.not_equal(centroids, prev_centroids).any() and iteration < max_iter:
        assigned_points = {}
        for i in range(cluster_count):
            assigned_points[i] = np.empty((1,input_len))

        for i in range(248):
            dists = euclidean(input[i], centroids)
            
            centroid_idx = np.argmin(dists)
            assignment[i] = centroid_idx
            arr = np.array([input[i]])
            assigned_points[centroid_idx] = np.concatenate((assigned_points[centroid_idx], arr), axis=0)

            # print(assigned_points[centroid_idx].shape)

                # assigned_points[centroid_idx].append(input[i])
        
        for i in range(cluster_count):
            assigned_points[i] = np.delete(assigned_points[i], 0, 0)
            # print(len(assigned_points[i]))
            # print(assigned_points[i].shape)
            # print(np.swapaxes(assigned_points[i], 0,1).shape)
            # print(np.mean(np.swapaxes(assigned_points[i], 0,1), axis=1))
            # print(np.mean(np.swapaxes(assigned_points[i], 0,1), axis=1).shape)
            prev_centroids[i] = centroids[i]
            centroids[i] = np.mean(np.swapaxes(assigned_points[i], 0,1), axis=1)

        # Push current centroids to previous, reassign centroids as mean of the points belonging to them
        # prev_centroids = centroids
        # centroids = [np.mean(np.swapaxes(cluster, 0,1), axis=1) for cluster in assigned_points]

        for i, centroid in enumerate(centroids):
            if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                centroids[i] = prev_centroids[i]
        iteration += 1
        print(iteration)
    
    return assignment

folder = "./plot_data/"
rest = np.load(folder + "rest_matrix.npy")
# memory = np.load(folder + "memory_matrix.npy")
# motor = np.load(folder + "motor_matrix.npy")
# math = np.load(folder + "math_matrix.npy")

coords = get_raw_coordinates()
coords = np.array(coords)
print(coords.shape)
assignment = k_means(coords, 8)
np.save("./k2-assignment.npy", assignment)