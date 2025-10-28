import argparse
import yaml
import fit_dcs.inverse.fit_homogeneous as fit_hom
from fit_dcs.forward.homogeneous_semi_inf import g1_norm
from fit_dcs.utils.data_loaders import weigh_g2
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Fit DCS data with homogeneous semi-infinite model")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="path to YAML config file"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print("Usage: fitdcs-fit -c path/to/config.yaml")
        return

    channels_idx = config["input"]["channels_idx"]
    beta_mode = config["beta_calculation"]["mode"]
    if beta_mode == "fixed":
        beta_calculator = fit_hom.BetaCalculator(
            mode="fixed",
            beta_fixed=config["beta_calculation"]["beta_fixed"]
        )
    elif beta_mode == "raw":
        beta_calculator = fit_hom.BetaCalculator(
            mode="raw",
            tau_lims=config["beta_calculation"]["tau_lims"]
        )
    elif beta_mode == "fit":
        beta_calculator = fit_hom.BetaCalculator(
            mode="fit",
            beta_init=config["beta_calculation"]["beta_init"],
            beta_bounds=config["beta_calculation"]["beta_bounds"],
        )
    else:
        raise ValueError(f"Unknown beta calculation mode: {beta_mode}")

    msd_model = fit_hom.MSDModelFit(
        model_name="brownian",
        param_init={"db": config["fitting"]["db_init"]},
        param_bounds={"db": config["fitting"]["db_bounds"]},
    )

    tau_lims_fit = config["fitting"]["tau_lims_fit"]
    g2_lim_fit = config["fitting"]["g2_lim_fit"]
    mua = config["experimental"]["mua"]
    musp = config["experimental"]["musp"]
    rho = config["experimental"]["rho"]
    n = config["experimental"]["n"]
    lambda0 = config["experimental"]["lambda0"]

    files = config["input"]["files"]
    for file in files:
        print(f"Processing file {file}")
        data = np.load(file)
        tau = data["tau"]
        g2_norm_multi = data["g2_norm"][:, channels_idx, :]
        countrate = data["countrate"][:, channels_idx]
        g2_norm = weigh_g2(g2_norm_multi, countrate)

        fitter = fit_hom.FitHomogeneous(
            g1_norm,
            msd_model,
            beta_calculator,
            tau_lims_fit,
            g2_lim_fit,
            mua=mua,
            musp=musp,
            rho=rho,
            n=n,
            lambda0=lambda0
        )
        fit_results = fitter.fit(tau, g2_norm)

        # Save results
        output_dir = Path(config["output"]["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(file).stem
        fit_results.to_csv(output_dir / f"{base_filename}_fit.csv")
        print(f"Saved fitted data to {output_dir / f'{base_filename}_fit.csv'}")
