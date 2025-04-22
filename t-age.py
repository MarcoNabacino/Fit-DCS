import utils.data_loaders as data_loaders
import pandas as pd
import os
import glob
import inverse.fit_homogeneous as fit_hom
import forward.homogeneous_semi_inf as hsi
import numpy as np

# Read general info file.
info_file = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Database_misure.xlsx"
info = pd.read_excel(info_file)
# Find indices of files to process based on "Eligible DCS" column
idx_files_to_process = info.index[info["Eligible DCS"] == "Yes"].tolist()

DCS_root = "W:/DATA/fNIRS_LAB/TRAJECTOR_AGE/Parma_Metaboliche/DCS/"
TRS_root = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Analisi/TRS/Output_processed/"
output_folder = "C:/Users/marco/OneDrive - Politecnico di Milano/Dottorato/Trajector-Age/Analisi/TimeCourses/"
for i_file in idx_files_to_process:
    # Read DCS data
    DCS_folder = info.loc[i_file, "DCS Folder"]
    DCS_path = os.path.join(DCS_root, DCS_folder)
    data_files_DCS = glob.glob(DCS_path + "*")
    data_files_DCS.sort()
    # Discard last file
    data_files_DCS = data_files_DCS[:-1]

    loader = data_loaders.DataLoaderALV(
        data_files_DCS,
        n_channels=4
    )
    loader.load_data()
    # Weight g2_norm by countrate
    g2_norm = data_loaders.weight_g2(loader.g2_norm, loader.countrate)
    # Discard noisy tau
    mask = loader.tau > 1e-7
    tau = loader.tau[mask]
    g2_norm = g2_norm[mask]

    # Read optical parameters
    subject = info.loc[i_file, "Subject"]
    time_point = info.loc[i_file, "Time Point"]
    exercise = info.loc[i_file, "Exercise"][0:3]
    TRS_file_name = f"{subject}_{time_point}_{exercise}.csv"
    TRS_path = os.path.join(TRS_root, TRS_file_name)
    TRS_data = pd.read_csv(TRS_path)
    mua = TRS_data["VarMua0Opt830"].values
    musp = TRS_data["VarMus0Opt830"].values

    #beta_calculator =  fit_hom.BetaCalculator(mode="fixed", beta_fixed=0.50)
    #beta_calculator = fit_hom.BetaCalculator(mode="raw", tau_lims=(1e-7, 2e-7))
    beta_calculator = fit_hom.BetaCalculator(mode="fit", beta_init=0.48, beta_bounds=(0.4, 0.6))
    msd_model = fit_hom.MSDModelFit(model_name="brownian", param_init={"db": 1e-8}, param_bounds={"db": (0, None)})
    #msd_model = fit_hom.MSDModelFit(model_name="hybrid", param_init={"db": 1e-8, "v_ms": 1e-6}, param_bounds={"db": (0, None), "v_ms": (0, None)})
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
        rho=2.5,
        n=1.4,
        lambda0=785
    )
    fitted_data = fitter.fit(plot_interval=0)

    # Add average counts between all channels to fitted_data DataFrame
    countrate = np.mean(loader.countrate, axis=-1)
    counts = countrate / info.loc[i_file, "Sample Frequency"]
    fitted_data["CountsDCS"] = counts

    # Add Iteration column at the beginning of the fitted_data DataFrame
    iteration = np.arange(0, len(fitted_data))
    fitted_data.insert(0, "Iteration", iteration)

    # Rename some columns in fitted_data DataFrame
    fitted_data.rename(columns={
        "db": "Db",
        "chi2": "Chi2DCS",
    }, inplace=True)

    # Merge fitted_data with TRS_data based on the "Iteration" column
    fitted_data = pd.merge(TRS_data, fitted_data, on="Iteration", how="left", validate="one_to_one")

    # Save fitted data to CSV file
    output_file_name = f"{subject}_{time_point}_{exercise}.csv"
    output_path = os.path.join(output_folder, output_file_name)
    fitted_data.to_csv(output_path, index=False)