"""Convert minc to nifti format."""

import os
import numpy as np
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image

name = "pd_icbm_normal_1mm_pn0_rf20"
minc = load(
    f"/mnt/data_drive/hrodrigo/mri_rician_noise/.nppy_testing/raw_brains/{name}.mnc"
)
basename = minc.get_filename().split(os.extsep, 1)[0]

affine = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

out = Nifti1Image(minc.get_fdata(), affine=affine)
save(out, name + ".nii.gz")
