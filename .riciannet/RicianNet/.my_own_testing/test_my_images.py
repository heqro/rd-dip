import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import caffe
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


def pad_image(img: np.ndarray, new_height: int, new_width: int):

    padding = np.zeros((new_height, new_width))
    img_height, img_width = img.shape
    padding[:img_height, :img_width] = img
    return padding


def psnr_with_mask(
    img_1: np.ndarray,
    img_2: np.ndarray,
    mask: np.ndarray,
    data_range=1.0,
):
    mask_size = (mask > 0).sum().item()
    mse = ((img_1 - img_2) ** 2 * mask).sum() / mask_size
    return 10 * np.log10(data_range**2 / mse)


mask = np.array(
    Image.open(
        "/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/images_exported_to_png/scalar_field.png"
    ).convert("L")
)
mask = mask / mask.max()
gt = np.array(
    Image.open(
        "/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/images_exported_to_png/slice_original.png"
    ).convert("L")
)
gt = gt / gt.max()

noise_factor = 0.15
sigma = noise_factor * gt.max()

gt = pad_image(gt, 256, 270)
mask = pad_image(mask, 256, 270)
f_real = gt + sigma * np.random.randn(gt.shape[-2], gt.shape[-1])
f_imag = sigma * np.random.randn(gt.shape[-2], gt.shape[-1])
f = np.sqrt(f_real**2 + f_imag**2)
residue = f - gt

plt.imsave(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/.my_own_testing/Contamined_{noise_factor}_Mine.png",
    f,
    cmap="gray",
)


prototext_path = "/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/Riciannet_deploy.prototxt"
weights_path = "/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/model/N15_complex_Brain1&2.caffemodel"

caffe.set_mode_cpu()
net = caffe.Net(prototext_path, weights_path, caffe.TEST)

result = net.forward(data=f.reshape(1, 1, 256, 270))
result_conv13 = result["conv13"].squeeze()


result_conv13_normalized = (result_conv13 - result_conv13.min()) / (
    result_conv13.max() - result_conv13.min()
)

print(f"PSNR is {psnr(gt, result_conv13)}")  # 27.76
print(f"PSNR is {psnr_with_mask(result_conv13, gt, mask)}")  # 24.14


plt.imsave(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/.my_own_testing/Denoised_{noise_factor}_RicianNet.png",
    result_conv13,
    cmap="gray",
)

plt.imsave(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/.my_own_testing/denoised_RicianNet_masked.png",
    (result_conv13 * mask),
    cmap="gray",
)

# plt.imsave(
#     f"/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/.my_own_testing/denoised_RicianNet_masked_normalized.png",
#     (result_conv13_normalized * mask),
#     cmap="gray",
# )
