#   Taken from Deep Blackbox Graph Matching: https://github.com/martius-lab/blackbox-deep-graph-matching.
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np


def locations_to_features_diffs(x_1, y_1, x_2, y_2):
    res = np.array([0.5 + 0.5 * (x_1 - x_2) / 256.0, 0.5 + 0.5 * (y_1 - y_2) / 256.0])
    return res


def build_graphs(P_np: np.ndarray, n: int, n_pad: int = None, edge_pad: int = None):

    A = delaunay_triangulate(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    edge_list = [[], []]
    features = []
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                edge_list[0].append(i)
                edge_list[1].append(j)
                features.append(locations_to_features_diffs(*P_np[i], *P_np[j]))

    if not features:
        features = np.zeros(shape=(0, 2))

    return np.array(edge_list, dtype=int), np.array(features)


def build_graphs_knn(P_np: np.ndarray, n: int, n_pad: int = None, edge_pad: int = None):

    A = k_nearest_neighbors(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    edge_list = [[], []]
    features = []
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                edge_list[0].append(i)
                edge_list[1].append(j)
                features.append(locations_to_features_diffs(*P_np[i], *P_np[j]))

    if not features:
        features = np.zeros(shape=(0, 2))

    return np.array(edge_list, dtype=int), np.array(features)



def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = np.ones((n, n)) - np.eye(n)
    else:
        try:
            d = Delaunay(P)
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print("Delaunay triangulation error detected. Return fully-connected graph.")
            print("Traceback:")
            print(err)
            A = np.ones((n, n)) - np.eye(n)
    return A


from sklearn.neighbors import NearestNeighbors
import numpy as np

def k_nearest_neighbors(point_set, k):
    """
    Compute the k-nearest neighbors for each point in a set.

    Parameters:
    - point_set: The set of points (numpy array or list of arrays).
    - k: The number of neighbors to find.

    Returns:
    - distances: Array containing distances to the k-nearest neighbors for each point.
    - indices: Array containing indices of the k-nearest neighbors for each point.
    """
    # Convert the point set to a numpy array if it's not already
    point_set = np.array(point_set)

    # Create a NearestNeighbors object
    neighbors_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    
    # Fit the model to the point set
    neighbors_model.fit(point_set)

    # Find the k-nearest neighbors for each point
    distances, indices = neighbors_model.kneighbors(point_set)

    return distances, indices

