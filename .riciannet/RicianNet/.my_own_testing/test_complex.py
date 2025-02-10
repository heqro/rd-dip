import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import caffe
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from PIL import Image

mat = scipy.io.loadmat("../testdata/Brain1.mat")
noiseSigma = 15
# mat['Img'] returns the same brain!
brain_not_normalized = mat["Img"]

brain = brain_not_normalized / np.abs(brain_not_normalized).max()
# Add Rician noise
level = noiseSigma * np.abs(brain).max() / 100
n1 = level * np.random.randn(*np.abs(brain).shape) + brain.real
n2 = level * np.random.randn(*np.abs(brain).shape) + brain.imag

brain_real = np.abs(brain)
plt.imsave("Brain1.png", brain_real, cmap="gray")

in_img = np.sqrt(n1**2 + n2**2)
# in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())

plt.imsave(f"Contaminada_{noiseSigma}.png", in_img, cmap="gray")


prototext_path = "/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/Riciannet_deploy.prototxt"
weights_path = "/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/model/N15_complex_Brain1&2.caffemodel"


caffe.set_mode_cpu()  # Use caffe.set_mode_gpu() if you want to use GPU

net = caffe.Net(prototext_path, weights_path, caffe.TEST)


result = net.forward(data=np.rot90(in_img)[None, None, ...])
result_conv13 = result["conv13"].squeeze()

plt.imsave(f"Denoised_{noiseSigma}.png", result_conv13, cmap="gray")


plt.imsave(f"Brain1.png", np.rot90(brain_real), cmap="gray")

residuo = result_conv13 - np.rot90(brain_real)
plt.imsave(f"Residuo_{noiseSigma}.png", residuo, cmap="gray")

print(psnr(result_conv13, np.rot90(brain_real), data_range=1.0))

# Load msk
mask = Image.open("Brain1_mask.png").convert("L")  # Convert to grayscale
mask = np.array(mask)


print(
    psnr(
        result_conv13[:256, :256] * mask,
        np.rot90(brain_real)[:256, :256],
        data_range=1.0,
    )
)


print(ssim(result_conv13, np.rot90(brain_real), data_range=1.0))
