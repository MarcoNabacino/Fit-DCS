from utils import data_loaders
import pandas as pd
import os
import numpy as np
import glob

DCS_root = "W:/DATA/fNIRS_LAB/TRAJECTOR_AGE/Parma_Metaboliche/DCS"
# Read general info file.
info_file = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Database_misure.xlsx"
info = pd.read_excel(info_file)
# Find indices of measurements to process based on "Eligible DCS" column
mask = info["Eligible DCS"] == "Yes"
idx_measurements_to_process = info.index[mask].tolist()

# Process each measurement
for i_meas in idx_measurements_to_process:
    # Get list of DCS files
    DCS_folder = info.loc[i_meas, "DCS Folder"]
    DCS_path = os.path.join(DCS_root, DCS_folder)
    print(f"Loading DCS data from {DCS_path}...")
    DCS_data_files = glob.glob(DCS_path + "*")
    DCS_data_files.sort()
    DCS_data_files = DCS_data_files[:-1]  # Discard last file
    # Load DCS data
    loader = data_loaders.DataLoaderALV(
        DCS_data_files,
        n_channels=4
    )
    loader.load_data()
    tau, g2_norm, countrate = loader.tau, loader.g2_norm, loader.countrate

    # Save data to .npz file
    output_file_name = info.loc[i_meas, "DCS Name"]
    output_file = os.path.join(DCS_path, output_file_name)
    np.savez(output_file, tau=tau, g2_norm=g2_norm, countrate=countrate)
    print(f"Saved {output_file_name} to {output_file}.")