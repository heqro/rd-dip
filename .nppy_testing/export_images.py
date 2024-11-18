from PIL import Image
import nibabel as nib
from nibabel import filebasedimages
import numpy as np


def save_img(
    img_array: filebasedimages.FileBasedImage, filename: str, is_scalar_field: bool
):
    data = img_array.get_fdata()[::-1, 127, :]
    modified_data = (
        np.where(data > 0.01, 255, 0).astype(np.uint8) if is_scalar_field else data
    )
    img = Image.fromarray(np.rot90(modified_data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(filename)


name_of_brain = "pd_icbm_normal_1mm_pn0_rf20.nii"  # 'subject04_t1w.nii, t1_icbm_normal_1mm_pn0_rf20.nii'
# Get original image
dumped_original_img = nib.load(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/segmented_brains/{name_of_brain}_orig.nii.gz"
)
save_img(
    dumped_original_img,
    "images_exported_to_png/slice_original_pd.png",
    is_scalar_field=False,
)

# Get scalar field
scalar_field = nib.load(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/segmented_brains/{name_of_brain}_scalar_field.nii.gz"
)
save_img(
    scalar_field, "images_exported_to_png/scalar_field_pd.png", is_scalar_field=True
)
