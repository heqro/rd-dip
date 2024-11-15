import numpy as np
import torch
import os

# Directory containing .npy files
npy_dir = "."
torch_dir = "."

# Ensure output directory exists
os.makedirs(torch_dir, exist_ok=True)

# Process all .npy files
for filename in os.listdir(npy_dir):
    if filename.endswith(".npy"):
        # Load the NumPy array
        np_array = np.load(os.path.join(npy_dir, filename))

        # Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(np_array)

        # Save as a PyTorch serialized file (.pt)
        torch_filename = os.path.splitext(filename)[0] + ".pt"
        torch.save(torch_tensor, os.path.join(torch_dir, torch_filename))

        print(f"Converted and saved: {filename} -> {torch_filename}")
