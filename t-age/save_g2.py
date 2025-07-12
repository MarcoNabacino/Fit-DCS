import utils.data_loaders as data_loaders
from utils.timetagger import get_correlator_architecture
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":
    # Input folder
    DCS_root = "W:/DATA/fNIRS_LAB/TRAJECTOR_AGE/Parma_Metaboliche/DCS/"

    # Read general info file.
    info_file = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Database_misure.xlsx"
    info = pd.read_excel(info_file)
    # Find indices of measurements to process based on "City" column
    mask = (info["City"] == "PV") & (info["Eligible DCS"] == "Yes")
    idx_measurements_to_process = info.index[mask].tolist()
    idx_measurements_to_process = idx_measurements_to_process[0:1] # For testing purposes, process only the first measurement

    # Process each measurement
    for i_meas in idx_measurements_to_process:
        subject = info.loc[i_meas, "Subject"]
        time_point = info.loc[i_meas, "TimePoint"]
        exercise = info.loc[i_meas, "Exercise"]
        print(f"Processing measurement {i_meas} ({subject}_{time_point}_{exercise})...")

        # Input file
        DCS_folder = info.loc[i_meas, "DCS Folder"]
        DCS_path = os.path.join(DCS_root, DCS_folder)
        DCS_name = info.loc[i_meas, "DCS Name"]
        DCS_data_file = DCS_path + DCS_name + ".ttbin"
        # Output file
        output_folder = DCS_path
        output_file_name = DCS_name
        output_file = os.path.join(output_folder, output_file_name)
        # PVY_010_Intm was split into 2 measurements
        if (subject == "PVY_010") & (time_point == "T0") & (exercise == "Int"):
            DCS_data_file = [DCS_path + name + ".ttbin" for name in ["PVY_010_Int", "PVY_010_Int2"]]

        print(f"Loading time tags from {DCS_data_file}...")
        # Load DCS data
        m = 2
        (p, s) = get_correlator_architecture(alpha=7, m=m, tau_max=1e-2, t0=1e-12)
        integration_time = 1 / info.loc[i_meas, "Sample Frequency"]
        tau_start = 1e-7
        loader = data_loaders.DataLoaderTimeTagger(
            DCS_data_file,
            integration_time=integration_time,
            channels=[1, 2, 3, 4],
            p=p,
            m=m,
            s=s,
            tau_start=tau_start,
        )
        loader.load_data(plot_interval=100)

        # Discard noisy tau channels
        mask = loader.tau > tau_start
        tau = loader.tau[mask]
        g2_norm = loader.g2_norm[mask, ...]
        countrate = loader.countrate

        # Save autocorrelation data to .npz file
        print(f"Saving autocorrelation data to {output_file}...")
        np.savez(output_file, tau=tau, g2_norm=g2_norm, countrate=countrate)

        print("Done.")
