import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("error") # Catch warnings in `extract_patches` as errors for testing

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from image_processing import extract_patches, get_patch_params

def is_valid_num_patches(num_patches, n, patch_size, patch_gap):
    correct_num_patches = ((n-patch_size)/patch_gap + 1)**2
    return num_patches == correct_num_patches

def main():
    test_values = np.logspace(4, 12, 12, base = 2).astype(int)
    for n in test_values:
        img_array = np.zeros(shape=(n,n))
        patches = extract_patches(img_array)  # No parameters passed, `extract_patches` will use `get_patch_params`. This can raise errors.
        patch_size, patch_gap = get_patch_params(n)
        assert is_valid_num_patches(patches.shape[0], n, patch_size, patch_gap), f"Wrong number of patches extracted for n={n}"
    print("test_patch_extraction passed.")

if __name__ == "__main__":
    main()
