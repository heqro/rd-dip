import os
import json


def load_full_table(json_files_path: list[str], JSON_DIRECTORY: str):
    table_data = []
    for file in json_files_path:
        with open(os.path.join(JSON_DIRECTORY, file), "r") as jsonfile:
            data = json.load(jsonfile)
        table_data.append(
            {
                "File": file,
                "PSNR Max": max(data["psnr_mask_log"]),
                "SSIM Max": max(data["ssim_mask_log"]),
            }
        )
    return table_data


def load_data_for(
    fidelity: str,
    regularizer: str,
    json_files: list[str],
    dip_rician_it: bool,
    use_smaller_rician_noise: bool,
):
    if regularizer != "":  # return many files
        return [
            filename
            for filename in json_files
            if f"_{fidelity}_{regularizer}" in filename
            and ("_itRician" in filename) == dip_rician_it
            and ("_itRician_0.05" in filename) == use_smaller_rician_noise
        ]
    # usually, return just one file
    suffix = (
        "_itRician_0.05"
        if use_smaller_rician_noise
        else "_itRician" if dip_rician_it else ""
    )

    return [
        filename
        for filename in json_files
        if filename.endswith(f"_{fidelity}{suffix}.json")
    ]


def load_tiered_table_column_data(filtered_files: list[str], JSON_DIRECTORY: str):
    import re

    sorted_files = sorted(
        filtered_files,
        key=lambda filename: (
            float(re.search(r":1e[+-]?\d:", filename).group(0)[1:-1])
            if re.search(r":1e[+-]?\d:", filename)
            else float("inf")
        ),
        reverse=True,
    )
    results = []
    for file in sorted_files:
        with open(f"{JSON_DIRECTORY}/{file}", "r") as jsonfile:
            data = json.load(jsonfile)
            results += [max(data["psnr_mask_log"])]
    return results


def filter_dct_files_by_dim(dct_files: list[str], dim: int):
    import re

    pattern = f":{dim}:"
    return [filename for filename in dct_files if re.search(pattern, filename)]


def load_tiered_table_columns(
    fidelities: list[str],
    regularizers: list[str],
    json_files: list[str],
    JSON_DIRECTORY: str,
):
    import pandas as pd

    df = pd.DataFrame()
    for fid in fidelities:
        only_fid = load_data_for(fid, "", json_files, False, False)
        for reg in regularizers:
            fid_and_reg = load_data_for(fid, reg, json_files, False, False)
            if reg == "Discrete_Cosine_Transform":
                fid_and_reg_3 = filter_dct_files_by_dim(fid_and_reg, 3)
                df[f"{fid}_DCT_3"] = load_tiered_table_column_data(
                    only_fid + fid_and_reg_3, JSON_DIRECTORY
                )
                fid_and_reg_5 = filter_dct_files_by_dim(fid_and_reg, 5)
                df[f"{fid}_DCT_5"] = load_tiered_table_column_data(
                    only_fid + fid_and_reg_5, JSON_DIRECTORY
                )
            else:
                df[f"{fid}_{reg}"] = load_tiered_table_column_data(
                    only_fid + fid_and_reg, JSON_DIRECTORY
                )
    return df
