import numpy as np
import warnings

def img_to_array(img):
    """
    Convert an image to grayscale and to a numpy array.

    Parameters:
    img : array-like, shape : (n, m)
        The image to convert.
    
    Returns:
    np.array, shape : (n, m)
        `img` converted to grayscale.
    
    """
    return np.array(img.convert('L'))

def get_patch_params(n: int):
    """
    Determine patches parameters (patch_size, patch_gap) of a square image of dimension (n, n).
    The parameters returned, ensures that `extract_patches` will extract every pixel in the image to a patch in a uniform way over the image.
    It also ensures no more than 10000 patches to be extracted.
    
    Parameters:
    n : Integer
        One of the square image dimensions.
    
    Returns:
    patch_size : integer
        The size of a patch in the image, patch will be of dimensions (patch_size, patch_size).
    patch_gap : Integer
        The pixel gap between each patch across any direction.
        For example, if a patch start in (0, 0) and patch_gap=4, the next patches will start at (4, 0) and (0, 4)

    """
    if n <= 100:
        patch_size = 5
        patch_gap = 1
    else:
        patch_size = np.floor(n / 15).astype(int)
        m = min(n**2, 10000)
        patch_gap = np.ceil((n-patch_size)/(np.sqrt(m)-1)).astype(int)
    indices = range(0, n-patch_size+1, patch_gap)
    # Patch size adjustment to ensure the whole image is extracted
    if indices[-1] + patch_size < n:
        patch_size = n - indices[-1]
    
    return patch_size, patch_gap

def extract_patches(img_array, patch_size, patch_gap=1):
    """
    Extract patches from a square image. 
    
    Parameters:
    img_array : array-like, shape : (n, n)
        The square image.
    patch_size : Integer
        The size of a patch in the image, patch will be of dimensions (patch_size, patch_size).
    patch_gap : Integer
        The pixel gap between each patch in any direction.
        For example, if a patch start in (0, 0) and patch_gap=4, the next patches will start at (4, 0) and (0, 4)
    
    Returns:
    np.array, shape : (n_patches, patch_size, patch_size)
        A 3D array such that every element is a patch extracted from `img_array`.
        Assuming `patch_size` and `patch_gap` fit the image (The full image can be extracted to patches), n_patches=((n-patch_size / patch_gap) + 1)**2

    """
    patches = []
    n = img_array.shape[0]
    k = patch_size
    if patch_gap > k:
        warnings.warn("When 'patch_gap' is larger than one of patch_size dimensions, the full image will not be extracted.")
    indices = range(0, n-patch_size+1, patch_gap)
    if indices[-1] + patch_size < n:
        warnings.warn(f"Couldn't extract patches for the full image. Consider using 'patch_size[0]'={n-indices[-1]} or changing 'patch_gap'")
    for i in indices:
        for j in indices:
            patch = img_array[i:i + k, j:j + k]
            patches.append(patch)
    return np.array(patches)