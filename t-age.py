import utils.data_loaders as data_loaders
import pandas as pd
import os
import glob
import inverse.fit_homogeneous as fit_hom
import inverse.mbl_homogeneous as mbl_hom
import forward.homogeneous_semi_inf as hsi
import numpy as np

# Input and output paths
DCS_root = "W:/DATA/fNIRS_LAB/TRAJECTOR_AGE/Parma_Metaboliche/DCS/"
TRS_root = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Analisi/TRS/Output_processed/"
output_folder = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Analisi/TimeCourses/"

# Read general info file.
info_file = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Database_misure.xlsx"
info = pd.read_excel(info_file)
# Find indices of measurements to process based on "Eligible DCS" column
mask = (info["Eligible DCS"] == "Yes") & (info["City"] == "PR")
idx_measurements_to_process = info.index[mask].tolist()

# Process each measurement
for i_meas in idx_measurements_to_process:
    subject = info.loc[i_meas, "Subject"]
    time_point = info.loc[i_meas, "TimePoint"]
    exercise = info.loc[i_meas, "Exercise"]
    print(f"Processing measurement {i_meas} ({subject}_{time_point}_{exercise})...")

    # Get list of DCS files
    DCS_folder = info.loc[i_meas, "DCS Folder"]
    DCS_path = os.path.join(DCS_root, DCS_folder)
    print(f"Loading DCS data from {DCS_path}...")
    DCS_data_files = glob.glob(DCS_path + "*")
    DCS_data_files.sort()
    DCS_data_files = DCS_data_files[:-1] # Discard last file
    # Load DCS data
    loader = data_loaders.DataLoaderALV(
        DCS_data_files,
        n_channels=4
    )
    loader.load_data()
    # Weigh g2_norm by countrate
    g2_norm = data_loaders.weigh_g2(loader.g2_norm, loader.countrate)
    # Discard noisy tau channels
    mask = loader.tau > 1e-7
    tau = loader.tau[mask]
    g2_norm = g2_norm[mask]

    # Read optical parameters from TRS file
    TRS_file_name = f"{subject}_{time_point}_{exercise}.csv"
    TRS_path = os.path.join(TRS_root, TRS_file_name)
    print(f"Loading TRS data from {TRS_path}...")
    TRS_data = pd.read_csv(TRS_path)
    mua = TRS_data["VarMua0Opt830"].values.copy()
    musp = TRS_data["VarMus0Opt830"].values.copy()
    # Some mua and musp values are nan, replace them with the previous value. Since in some cases there are multiple
    # consecutive nan values, we need to use a loop to replace them.
    for j in range(1, len(mua)):
        if np.isnan(mua[j]):
            mua[j] = mua[j - 1]
        if np.isnan(musp[j]):
            musp[j] = musp[j - 1]

    # Forward model parameters
    rho = 2.5 # cm
    n = 1.4 # refractive index
    lambda0 = 785 # nm

    # Fit DCS data
    print("Fitting DCS data...")
    beta_calculator = fit_hom.BetaCalculator(mode="fit", beta_init=0.48, beta_bounds=(0.4, 0.6))
    msd_model = fit_hom.MSDModelFit(model_name="brownian", param_init={"db": 1e-8}, param_bounds={"db": (0, None)})
    fitter = fit_hom.FitHomogeneous(
        tau,
        g2_norm,
        hsi.g1_norm,
        msd_model,
        beta_calculator,
        tau_lims_fit=(3e-7, 1e-2),
        g2_lim_fit=1.13,
        mua=mua,
        musp=musp,
        rho=rho,
        n=n,
        lambda0=lambda0
    )
    fitted_data = fitter.fit(plot_interval=0)
    # Rename columns in fitted_data DataFrame
    fitted_data.rename(columns={"db": "Db", "chi2": "Chi2DCS", "r2": "R2DCS"}, inplace=True)
    # Add average counts between all channels to fitted_data DataFrame
    countrate = np.mean(loader.countrate, axis=-1)
    sampling_frequency = info.loc[i_meas, "Sample Frequency"]
    counts = countrate / sampling_frequency
    fitted_data["CountsDCS"] = counts
    # Add columns of fitted_data to TRS_data DataFrame by joining on indices
    output_df = TRS_data.join(fitted_data)

    # Save fitted data to CSV file
    output_file_name = f"{subject}_{time_point}_{exercise}.csv"
    output_path = os.path.join(output_folder, output_file_name)
    print(f"Saving fitted data to {output_path}...")
    output_df.to_csv(output_path, index=False)


    # MBL analysis
    # Find baseline g2_norm and parameters by averaging the last seconds of measurements before the first tag.
    print("Performing MBL analysis...")
    first_tag_index = int(info.loc[i_meas, "Tag 1"])
    baseline_time_mbl = 20 # s
    baseline_delta_tag = int(baseline_time_mbl * sampling_frequency)
    g2_norm_0 = np.mean(g2_norm[:, first_tag_index - baseline_delta_tag:first_tag_index], axis=-1)
    db0 = np.mean(fitted_data["Db"].values[first_tag_index - baseline_delta_tag:first_tag_index])
    mua0 = np.mean(mua[first_tag_index - baseline_delta_tag:first_tag_index])
    musp0 = np.mean(musp[first_tag_index - baseline_delta_tag:first_tag_index])
    # Perform MBL analysis.
    msd_model_mbl = mbl_hom.MSDModelMBL("brownian", db0)
    mbl_analyzer = mbl_hom.MBLHomogeneous(
        tau,
        g2_norm,
        g2_norm_0,
        hsi.d_factors,
        msd_model_mbl,
        mua,
        musp,
        mua0=mua0,
        musp0=musp0,
        rho=rho,
        n=n,
        lambda0=lambda0
    )
    db_mbl = mbl_analyzer.fit()

    # Save MBL analysis results to a separate .csv file
    output_file_name_mbl = f"{subject}_{time_point}_{exercise}_mbl.csv"
    output_path_mbl = os.path.join(output_folder, output_file_name_mbl)
    print(f"Saving MBL analysis results to {output_path_mbl}...")
    # First row is tau transposed, subsequent rows are Db_mbl transposed
    # (each row is an iteration and each column a different tau)
    db_mbl_output = np.vstack((tau.T, db_mbl.T))
    np.savetxt(output_path_mbl, db_mbl_output, delimiter=",")

    print("Done.")
