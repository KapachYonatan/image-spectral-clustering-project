import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from image_processing import extract_patches, get_patch_params

def compute_kernel_matrix(X, k_neighbors):
    """
    Compute the gaussian kernel matrix K with local bandwidth values. K is a matrix where K_ij = exp(-||X_i - X_j||^2 / sigma_i*sigma_j).
    sigma_i is the k-th nearest neighbor of X_i.

    Parameters:
    X : array-like, shape : (n, m)
        The matrix that the kernel will be computed over.
    k : Integer
        Number of neighbors to consider for each sigma value. 

    Returns:
    np.array, shape : (n, n)
        The kernel matrix.

    """
    # Compute the distances of the kth nearest neighbor for each row in X
    nn = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    sigma = distances[:, -1]
    sigma[sigma == 0] = min(1e-7, np.min(sigma[sigma != 0])) # Smooth sigma values that are 0

    # Construct the kernel gaussian matrix
    dists = cdist(X, X, 'sqeuclidean')
    S = np.tile(1 / sigma, (len(sigma), 1))
    S_2 = np.multiply(S, S.T) # Element wise multiplication to get S_2_ij = 1 / (sigma_i * sigma_j)
    K = np.exp(-(np.multiply(dists, S_2)))
        
    return K

def compute_laplacian(K):
    """
    Compute the normalized Laplacian matrix L of the symmetric matrix K.
    L = I - D^-1/2 * K D^-1/2 where D is a diagonal matrix with D_ii is the sum of the i-th row/column.

    Parameters:
    K : array-like, shape : (n, n)
        The matrix that the normalized Laplacian will be computed over. K is assumed to be symmetric.

    Returns:
    np.array, shape : (n, n)
        The normalized Laplacian matrix.

    """
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(K, axis=1)))
    L = np.eye(K.shape[0]) - D_inv_sqrt @ K @ D_inv_sqrt
    return L

def k_smallest_eig(L, k, skip_first=True):
    """
    Extracts the k smallest eigen values and vectors of the symmetric matrix L. If 'skip_first', it skips the first eigen value.
    
    Parameters:
    L : array-like, shape : (n, n)
        The symmetric matrix to extract the eigen values from.
    k : Integer
        The number of eigen values/vectors to extract
    skip_first : bool, default=True
        If True, it ignores the first eigen value (smallest). Useful when inputting Laplacian matrix to ignore the trivial 0 eigen value.

    Returns:
    np.array, shape : (k, )
        k smallest eigen values of L.
    np.array, shape : (n, k)
        Corresponding eigen vectors.
    
    """
    if skip_first:
        eig_values, eig_vectors = eigsh(L, k=k+1, which='SM')
        return eig_values[1:(k+1)], eig_vectors[:, 1:(k+1)]
    return eigsh(L, k=k, which='SM')

def spectral_clustering(X, n_clusters, k_neighbors=20, random_state=42):
    """
    Perform spectral clustering over the rows of a given matrix X.

    Parameters:
    X : array-like, shape : (n, m)
        The matrix.
    n_clusters : Integer
        The number of clusters desired.
    k_neighbors : Integer, default=20
        The parameter that will be passed to `compute_kernel_matrix`
    random_state : Integer, default=42
        The parameter that will be passed to `KMeans`

    Returns:
    np.array, shape : (n, )
        The cluster labels of the rows of matrix X. This vector contains `n_clusters` unique values. 
    
    """
    print("Computing kernel matrix...")
    K = compute_kernel_matrix(X, k_neighbors)
    print("Extracting eigen vectors for spectral embedding...")
    L = compute_laplacian(K)
    eig_values, U = k_smallest_eig(L, k=n_clusters, skip_first=True)
    print("Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(U)
    
    return kmeans.labels_

def img_spectral_clustering(img_array, n_clusters, k_neighbors=20, random_state=42):
    """
    Extract patches from an image (has to be a square image) and performs spectral clustering over the patches.
    The labels are reshaped into an image which can be used for image segmentation purposes of the original image.

    Parameters:
    img_array : array-like, shape : (n, n)
        A grayscale square image.
    n_clusters : Integer
        The number of clusters (segments in the image) desired.
    k_neighbors : Integer, default=20
        The parameter that will be passed to `compute_kernel_matrix`
    random_state : Integer, default=42
        The parameter that will be passed to `KMeans`

    Returns:
    np.array, shape : (n_patches, n_patches)
        The cluster labels for each patch extracted from `img_array`, reshaped into a square image array. This is the segmented image. 
    
    """
    if img_array.shape[0] != img_array.shape[1]:
        raise ValueError("Image dimensions should be equal, of the shape: (a, a)")
    print("Extracting image patches...")
    patch_size, patch_gap = get_patch_params(img_array.shape[0])
    patches = extract_patches(img_array, patch_size, patch_gap=patch_gap)
    n_patches = len(patches)

    # Create the matrix X where each feature is a pixel of a patch
    X = patches.reshape(n_patches, patch_size**2)

    # Spectral clustering
    labels = spectral_clustering(X, n_clusters, k_neighbors, random_state)

    # Reshape to return an image array
    reshape_dim = np.sqrt(len(labels)).astype(int)

    return labels.reshape((reshape_dim, reshape_dim))