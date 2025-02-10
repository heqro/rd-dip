import itertools


p = 1.0
brain_idx = 1
brain_std = "0.15"
n = 6
# Combinations go here
archs = ["UNet"]  # ["UNet", "AttentiveUNet"]
fidelities = [
    "Rician_Norm",
]  # ["Gaussian", "Rician", "Rician_Norm"]
coefs = [
    1e0,
    1e1,
    # 1e2,
    # 1e3,
    # 1e4,
    # 1e5,
    # 1e6,
    # 9e-1,
    # 8e-1,
    # 7e-1,
    # 6e-2,
    # 5e-1,
    # 4e-1,
    # 3e-1,
    # 2e-1,
    1e-1,
    # 9e-2,
    # 8e-2,
    # 7e-2,
    # 6e-2,
    # 5e-2,
    # 4e-2,
    # 3e-2,
    # 2e-2,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
]  # 'escalar a mano sabiendo qué rangos son buenos'
regularizers = [
    # ("Total_Variation", f"1.0:{p}"),
    ("Discrete_Cosine_Transform", f"3:{p}"),
    # ("Discrete_Cosine_Transform", f"5:{p}"),
]
dip_noise_perturbations = ["Gaussian"]  # Rician
dip_noise_stds = [0.05, 0.15]  # 0.15
lr_list = [
    # 1e-2,
    "1e-3",
    # "3e-3",
    # "1e-3",
]

lists = [[] for _ in range(n)]
for idx, (arch, fid, reg, coef, dip_noise_type, dip_noise_std, lr) in enumerate(
    itertools.product(
        archs,
        fidelities,
        regularizers,
        coefs,
        dip_noise_perturbations,
        dip_noise_stds,
        lr_list,
    )
):
    fid_com = f"{fid}:1.0" if fid == "Gaussian" else f"{fid}:1.0:{brain_std}"
    reg_com = f"{reg[0]}:{coef}:{reg[1]}"
    command = f"python _denoise_complex_brain.py --model {arch} --fidelities {fid_com} --regularizers {reg_com} --dip_noise_type {dip_noise_type} --dip_noise_std {dip_noise_std} --lr {float(lr)}"
    command += f" --tag {arch}_Std{brain_std}_{fid}_{reg[0]}:{coef}:{reg[1]}_DIPNoise_{dip_noise_type}_DIPStd{dip_noise_std}_lr{lr}"
    lists[idx % n].append(command)
filename = "all_commands_AtteNet.sh"
with open(filename, "w") as f:
    f.write("#!/bin/bash\n\n")  # Add the shebang for bash scripts
    for lst in lists:
        # Join commands in each list with semicolons and append '&'
        f.write(" && ".join(lst) + " &\n")
import os

os.chmod(filename, 0o755)  # Set execute permissions for the file

print(f"Single .sh file '{filename}' created with parallel execution commands.")
