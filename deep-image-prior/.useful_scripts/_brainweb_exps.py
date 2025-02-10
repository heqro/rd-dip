import itertools


p = 1.0

brain_std = "0.20"
n = 6
# Combinations go here
archs = ["UNet"]  # ["UNet", "AttentiveUNet"]
fidelities = [
    "Gaussian",
    # "Rician",
    "Rician_Norm",
]  # ["Gaussian", "Rician", "Rician_Norm"]
coefs = [
    5e-1,
    3e-1,
    7e-1,
    1e0,
    # 3e0,
    # 1e-4,
    # 1e-5,
    # 1e-6,
    # 1e-7,
]  # 'escalar a mano sabiendo qué rangos son buenos'
regularizers = [
    # ""
    # ("Total_Variation", f"1.0:{p}"),
    ("Discrete_Cosine_Transform", f"3:{p}"),
    # ("Discrete_Cosine_Transform", f"5:{p}"),
    # ("Kirsch_Filter", f"1.0:{p}"),  # Prewitt
]
dip_noise_perturbations = ["Gaussian", "Rician"]  # Gaussian
dip_noise_stds = [0.15]  # 0.15, 0.05
lr_list = [
    # 1e-2,
    # "5e-5",
    # "1e-5",
    # "3e-5",
    # "3e-3",
    "5e-4",
]

lists = [[] for _ in range(n)]
n_times = 3
for idx, (arch, fid, reg, coef, dip_noise_type, dip_noise_std, lr, time) in enumerate(
    itertools.product(
        archs,
        fidelities,
        regularizers,
        coefs,
        dip_noise_perturbations,
        dip_noise_stds,
        lr_list,
        list(range(n_times)),
    )
):
    fid_com = f"{fid}:1.0" if fid == "Gaussian" else f"{fid}:1.0:{brain_std}"
    command = f"python _denoise_brainweb_brain.py --noise_std {brain_std} --model {arch} --fidelities {fid_com} --dip_noise_type {dip_noise_type} --dip_noise_std {dip_noise_std} --lr {float(lr)} --subject $1"
    if reg != "":
        command += f" --regularizers {reg[0]}:{coef}:{reg[1]}"

    lists[idx % n].append(command)
filename = "allcommands_dct.sh"
with open(filename, "w") as f:
    f.write("#!/bin/bash\n\n")  # Add the shebang for bash scripts
    f.write(
        """
if [ $# -eq 0 ]; then
    echo "Usage: $0 <numeric_parameter>"
    exit 1
fi

subj=$1

            
"""
    )
    for lst in lists:
        # Join commands in each list with semicolons and append '&'
        f.write(" && ".join(lst) + " &\n")
    f.write("wait\n")
import os

os.chmod(filename, 0o755)  # Set execute permissions for the file

print(f"Single .sh file '{filename}' created with parallel execution commands.")
