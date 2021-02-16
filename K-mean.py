import numpy as np
import scipy.io
import matplotlib.image as img
import matplotlib.pyplot as plt


def get_shortest_distance_index(X, centroids):
    # X and centroids should have same columns
    m = X.shape[0]
    K = centroids.shape[0]
    distances = np.zeros([m, K])
    for i in range(K):
        distances[:, i] = np.sum((X - centroids[i, :]) ** 2, axis=1)
    indexes = distances.argmin(axis=1)
    return indexes


def compute_centroids(X, K, indexes):
    cols = X.shape[1]
    centroids = np.zeros([K, cols])
    for i in range(K):
        centroids[i, :] = np.mean(X[indexes == i, :], axis=0)
    return centroids


def run_K_mean(X, initial_centroids, max_iter):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    for i in range(max_iter):
        index = get_shortest_distance_index(X, centroids)
        centroids = compute_centroids(X, K, index)

    return centroids


def random_initial_centroids(X, K):
    n = X.shape[1]
    random_index = np.random.permutation(X.shape[0])
    centroids = X[random_index[:K], :]
    return centroids


img_filename = "bird_small.png"
# 128 pixel * 128  pixel * 3 color
image = img.imread(img_filename)
# reshape image to 128*128  * 3
image_size1 = image.shape[0]
image_size2 = image.shape[1]
reshape_image = image.reshape([image_size1 * image_size2, -1])
K = 16
init_centroids = random_initial_centroids(reshape_image, K)
max_iter = 10
centroids = run_K_mean(reshape_image, init_centroids, max_iter)
index = get_shortest_distance_index(reshape_image, centroids)
recover_image = centroids[index, :]
recover_image = recover_image.reshape([image_size1, image_size2, -1])

# plt.imshow(image)
plt.imshow(recover_image)
plt.show()