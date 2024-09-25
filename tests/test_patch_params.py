import sys
import os
import numpy as np

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from image_processing import get_patch_params

def is_valid_range_patch_params(n, patch_size, patch_gap):
    return min(patch_size, patch_gap) > 0 and max(patch_size, patch_gap) < n 

def main():
    test_values = np.logspace(1, 14, 14, base = 2).astype(int)
    print(test_values)
    for n in test_values:
        patch_size, patch_gap = get_patch_params(n)
        assert patch_size >= patch_gap, f"`patch_size` is smaller than `patch_gap` with values: n={n}, patch_size={patch_size}, patch_gap={patch_gap}"
        assert is_valid_range_patch_params(n, patch_size, patch_gap), f"Params not in valid range with values: n={n}, patch_size={patch_size}, patch_gap={patch_gap}"
    print("test_patch_params passed.")

if __name__ == "__main__":
    main()