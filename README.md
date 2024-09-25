# Image Segmentation using Spectral Clustering

This project implements an image segmentation algorithm using spectral clustering techniques. It processes images by extracting patches, applying spectral clustering, and visualizing the results.

## Project Overview

The main components of this project are:

1. Image processing and patch extraction
2. Spectral clustering implementation
3. Visualization of clustering results

## Spectral Clustering Algorithm

This project implements the spectral clustering algorithm as described in the paper:

> Andrew Y. Ng, Michael I. Jordan, and Yair Weiss. "On Spectral Clustering: Analysis and an algorithm." In Advances in Neural Information Processing Systems (NeurIPS), 2002. [Link to paper](https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)

The algorithm works as follows:

1. Construct a similarity graph between data points
2. Compute the normalized Laplacian matrix
3. Find the k smallest eigenvectors of the Laplacian
4. Cluster the rows of the eigenvector matrix using K-means

My implementation adapts this algorithm for image segmentation by treating image patches as data points.

## Patch Extraction

The patch extraction process is a crucial step in the image segmentation pipeline:

1. The input image is divided into small, overlapping patches. For this implementation, the input image ratio has to be 1:1 (square image).
2. Each patch is a square section of the image.
3. Patches are extracted with a sliding window approach, moving across the image with a specified step size.
4. The extracted patches capture local image features and textures.
5. These patches become the data points for the spectral clustering algorithm.

The function `get_patch_params` in image_processing.py file can determine the patch dimensions (*patch_size*) and step size between patches (*patch_gap*). It ensures every pixel will be extracted to some patch and limits the total number of patches extracted (for computational reasons). 
Denote $k$ as the *patch_size* so that each patch is of dimensions $\left(k,k\right)$ and denote $p$ as the *patch_gap*. The number of patches that will be extracted from an image of dimensions $\left(n,n\right)$ is
```math
n\_patches=\left(\frac{n-k}{p}+1\right)^{2}
```
We try to aim for $n\\_patches$ to be $10000$ maximum. It is also needed to ensure $k\geq p$, otherwise some pixels will be skipped. Whenever $n>100$, the function

## Usage

A demonstration usage can be seen in the notebook file.
