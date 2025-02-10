import matplotlib.pyplot as plt
from _utils import load_gray_image
from PIL import Image
import numpy as np

# Load the image (assuming load_gray_image function is used)
# imtitle = "Im7-RicianNormDCT.png"
# imtitle = "DenoisingofMagneticResonanceImageswithDeepNeuralRegularizerDrivenbyImagePrior-slice4.png"
# imtitle = "Im7-Std0.15.png"
# imtitle = "Im7-Gaussian.png"
# imtitle = "Im7-RicianNet.png"
# imtitle = "/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/images_exported_to_png/slice_original_DenoisingofMagneticResonanceImageswithDeepNeuralRegularizerDrivenbyImagePrior-slice2.png"
imtitle = "/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior/.brainweb_test_data/im_5/Std0.15.png"
# imtitle = "/mnt/data_drive/hrodrigo/mri_rician_noise/deep-image-prior/bd853dc4-80f9-4e80-a97e-f108eb8f7ec8.png"

img = load_gray_image(imtitle, is_mask=False).numpy()[0]

# Define the regions of interest (ROIs)
rois_gaussian_motivation_im5 = [
    (15, 30, 48, 48),
    (100, 170, 48, 48),
    (140, 80, 48, 48),
]  # Coordinates (x, y, width, height)
rois = rois_gaussian_motivation_im5
# Define the regions of interest (ROIs)
# rois = [
#     (70, 35, 48, 48),
#     (100, 130, 48, 48),
#     (140, 80, 48, 48),
# ]  # Coordinates (x, y, width, height)


# Create a figure for displaying the ROIs in a row
fig, axs = plt.subplots(1, len(rois))

# Loop through ROIs and plot each
for i, (ax, (x, y, w, h)) in enumerate(zip(axs, rois)):
    roi = img[y : y + h, x : x + w]  # Crop the region
    ax.imshow(roi, cmap="gray")
    ax.axis("off")  # Remove axes for clean display

    # Draw a red rectangle around the ROI image
    ax.add_patch(
        plt.Rectangle(
            (-0.7, -0.5),
            roi.shape[1] + 0.15,
            roi.shape[0] - 0.35,
            edgecolor="red",
            facecolor="none",
            lw=1,
        ),
    )
    fig.patch.set_edgecolor("red")
    fig.patch.set_linewidth(2)
    fig.patch.set_zorder(10)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"{imtitle}_rois.png", bbox_inches="tight", pad_inches=0)


# Optionally display the original image with ROI rectangles
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)  # Preserve intensity when displaying
ax.axis("off")

# Draw rectangles on the original image
for x, y, w, h in rois:
    rect = plt.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", lw=2)
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig(f"{imtitle}_with_rois.png", bbox_inches="tight", pad_inches=0)
