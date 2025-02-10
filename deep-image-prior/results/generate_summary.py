import os
import json
import pandas as pd


def generate_summary(
    jsons_folder: str, output_folder: str, output_filename: str, target_std: float
):

    json_files = [f for f in os.listdir(jsons_folder) if f.endswith(".json")]
    list_of_reports = []

    for json_file in json_files:
        with open(f"{jsons_folder}/{json_file}", "r") as file:
            data = json.load(file)
            if data["image_noise_std"] != target_std:
                continue
            else:
                print(f"Considering {json_file}")
            dip_config = data["dip_config"]

            # Expand dip_config dictionary fields into separate columns
            expanded_config = {
                f"dip_config_{key}": value for key, value in dip_config.items()
            }
            addends = []
            for key in data["loss_log"]["addends"].keys():
                addends += [f'{key}*{data["loss_log"]["addends"][key]["coefficient"]}']
            report = {
                **expanded_config,  # Add the expanded dip_config fields
                "addends": addends,
                "brain_noise_std": data["image_noise_std"],
                "max_psnr_entire": max(data["psnr_entire_image_log"]),
                "max_psnr_mask": max(data["psnr_mask_log"]),
                "max_ssim_entire": max(data["ssim_entire_image_log"]),
                "max_ssim_mask": max(data["ssim_mask_log"]),
                "exit_code": data["exit_code"],
                "lr": data["optimizer"]["lr"],
                "filename": json_file,
            }
            list_of_reports.append(report)

    df = pd.DataFrame(list_of_reports)
    df.to_csv(f"{output_folder}/{output_filename}", index=False, sep=";")


target_std = 0.20
for i in range(1, 2):
    generate_summary(f"im_{i}/def_jsons", f"im_{i}", "summary_Std020.csv", target_std)
