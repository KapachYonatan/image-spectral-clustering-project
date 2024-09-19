import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, title=None):
    """
    Convenience function to plot an image.
    
    Parameters:
    image : array-like, shape : (n, m)
        An image to plot.
    title : str
        A title for the figure.
    
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_graph_node_degrees(K):
    """
    Plot the node degrees distribution of the adjacency matrix K. K is assumed to be symmetric (the case of undirected graph).
    
    Parameters:
    K : array-like, shape : (n, n)
        A symmetric adjacency matrix to plot its degrees distribution.
    
    """
    # Analyze the graph
    degrees = np.sum(K, axis=1)
    plt.figure(figsize=(10, 5))
    plt.hist(degrees, bins=50)
    plt.title("Distribution of Node Degrees")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

def plot_eigen_values(eig_values):
    """
    Convenience function to plot eigen values. Generates a scatter plot.
    
    Parameters:
    eig_values : array-like, shape : (n, )
        Array of values to plot.
    
    """
    plt.plot(eig_values, 'o')
    plt.title("Eigen values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

def plot_eigen_vectors_images(eig_vectors):
    """
    Convenience function to plot eigen vectors as images. A image plot will be generated for each eigen vector (column) inputted.
    
    Parameters:
    eig_vectors : array-like, shape : (n, m)
        A 2D array where each column is an eigen vector to plot.
    
    """
    reshaped_imgs = []
    reshape_dim = np.sqrt(eig_vectors.shape[0]).astype(int)
    for i in range(eig_vectors.shape[1]):
        vec = eig_vectors[:, i]
        reshaped_imgs.append(vec.reshape((reshape_dim, reshape_dim)))
    for i, img in enumerate(reshaped_imgs):
        plt.title(f"Eigen vector {i}")
        plt.imshow(img)
        plt.show()

def plot_clustering_results(original, segmented, n_clusters):
    """
    Plot original image alongside segmentation results.

    Parameters:
    original : array-like, shape : (n, m)
        The original image.
    segmented : array-like, shape : (k, p)
        The segmented image.
    n_clusters : Integer
        The number of clusters to present in the figure title.
    
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(segmented)
    ax2.set_title(f"Segmented Image (n_clusters={n_clusters})")
    ax2.axis('off')
    
    plt.show()
