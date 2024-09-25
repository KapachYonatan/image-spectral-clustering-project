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
#### `get_patch_params` behavior
Let $k$ be the patch_size, defining patches of dimensions $(k,k)$, and $p$ be the patch_gap. For an image of dimensions $(n,n)$, the number of extracted patches is:
```math
n\_patches = \left(\frac{n-k}{p}+1\right)^2
```
We aim to keep $n\\_patches \leq 10000$ and ensure $k \geq p$ to avoid skipping pixels. The function sets $k = \max(1, \lfloor n/15 \rfloor)$ and adjusts $p$ accordingly. To target $m = \min(n^2, 10000)$ patches, we derive $p$ from:
```math
m = \left(\frac{n-k}{p}+1\right)^2 \Rightarrow p = \left\lceil \frac{n-k}{\sqrt{m}-1}\right\rceil
```
To ensure the last patch in each row/column aligns with the image edge, the function may further adjust $k$. This approach balances comprehensive coverage with computational efficiency, adapting to various image sizes while maintaining consistent patch extraction.
## Usage

A demonstration usage can be seen in the notebook file.
