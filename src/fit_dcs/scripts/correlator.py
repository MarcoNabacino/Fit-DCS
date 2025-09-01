import argparse
import yaml
from fit_dcs.utils.data_loaders import DataLoaderTimeTagger
from fit_dcs.utils.timetagger import get_correlator_architecture
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Process time tagger data and compute autocorrelations.")
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
        print("Usage: fitdcs-corr -c path/to/config.yaml")
        return

    m = int(config["correlator"]["architecture"]["m"])
    alpha = int(config["correlator"]["architecture"]["alpha"])
    tau_max = float(config["correlator"]["architecture"]["tau_max"])
    (p, s) = get_correlator_architecture(alpha=alpha, m=m, tau_max=tau_max, t0=1e-12)
    integration_time = float(config["correlator"]["integration_time"])
    channels = config["correlator"]["channels"]
    tau_start = float(config["correlator"]["architecture"]["tau_start"])

    files = config["input"]["files"]
    for file in files:
        print(f"Processing file {file}")
        loader = DataLoaderTimeTagger(
            file,
            integration_time=integration_time,
            channels=channels,
            p=p,
            m=m,
            s=s,
            tau_start=tau_start
        )
        loader.load_data()

        # Discard initial tau values below tau_start (they are zero)
        mask = loader.tau > tau_start
        tau = loader.tau[mask]
        g2_norm = loader.g2_norm[mask, ...]
        countrate = loader.countrate

        # Save results
        output_dir = Path(config["output"]["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(file).stem
        np.savez(
            output_dir / f"{base_filename}_corr.npz",
            tau=tau,
            g2_norm=g2_norm,
            countrate=countrate
        )
        print(f"Saved correlation data to {output_dir / f'{base_filename}_corr.npz'}")