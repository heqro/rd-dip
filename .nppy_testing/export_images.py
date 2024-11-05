from PIL import Image
import nibabel as nib
from nibabel import filebasedimages
import numpy as np

def save_img(img_array: filebasedimages.FileBasedImage, filename: str, is_scalar_field: bool):
    data = img_array.get_fdata()[100,::-1,:]
    modified_data = np.where(data > 0.01, 255, 0).astype(np.uint8) if is_scalar_field else data
    padded_data = np.pad(modified_data, ((0,0), (7, 7)), mode='constant', constant_values=0)
    img = Image.fromarray(padded_data)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(filename)


# Get original image
dumped_original_img=nib.load('/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/segmented_brains/subject04_t1w.nii_orig.nii.gz')
save_img(dumped_original_img, 'images_exported_to_png/slice_original.png', is_scalar_field=False)

# Get scalar field
scalar_field=nib.load('/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/segmented_brains/subject04_t1w.nii_scalar_field.nii.gz')
save_img(scalar_field, 'images_exported_to_png/scalar_field.png', is_scalar_field=True)