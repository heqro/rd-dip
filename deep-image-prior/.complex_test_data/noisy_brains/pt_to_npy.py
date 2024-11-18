import os
import torch
import numpy as np

# List all files in the current directory
for filename in os.listdir("."):
    # Process only .pt files
    if filename.endswith(".pt"):
        # Load the .pt file
        tensor = torch.load(filename)

        # Convert to a NumPy array
        numpy_array = tensor.numpy()

        # Define the new filename with .npy extension
        npy_filename = filename.replace(".pt", ".npy")

        # Save as .npy file
        np.save(npy_filename, numpy_array)
        print(f"Converted {filename} to {npy_filename}")
