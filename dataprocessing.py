import numpy as np
from sklearn.cluster import SpectralClustering

def PCA(matrix, k):
    """Returns dimension reduced on dataset using k top eigenvalues/vectors of the matrix, and an array of the proportion of the total variance
    explained by each of the k principal components."""
    # Center the data matrix
    matrix = matrix - np.mean(matrix, axis=0)

    # Perform SVD on data matrix
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Get principal components, explained variances
    principal_components, variances = U @ np.diag(S), ((S ** 2)/(matrix.shape[0] - 1))
    explained_variances = variances/np.sum(variances)
    return principal_components[:,:k], explained_variances[:k]

def spectral_clustering(returnmatrix, clusters = 10, clustering_method = 'cluster_qr', similarity_transform = 'rbf'):
    """Returns list of arrays of row indices that are clustered together"""
    clustering = SpectralClustering(n_clusters=clusters,
        assign_labels=clustering_method,
        affinity = similarity_transform,
        random_state=0).fit(returnmatrix)
    return [np.where(clustering.labels_ == i)[0] for i in range(clustering.n_clusters)]
    
    