"""
 Fit-DCS: A Python toolbox for Diffuse Correlation Spectroscopy analysis
 Copyright (C) 2025  Marco Nabacino

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from fit_dcs.utils.data_loaders import DataLoaderTimeTagger
from fit_dcs.utils.timetagger import get_correlator_architecture
from pathlib import Path
import numpy as np
import os


def choose_worker_counts(n_channels, n_files, reserve_cores):
    # Determine number of available CPU cores
    n_cpu = os.cpu_count() or 1
    n_cpu = max(1, n_cpu - reserve_cores)

    # Choose number of inner workers, aiming for at most 4.
    max_inner_workers = min(n_channels, 4, n_cpu)
    # Choose number of outer workers to use remaining cores
    max_outer_workers = max(1, n_cpu // max_inner_workers)
    # Don't use more outer workers than files
    max_outer_workers = min(max_outer_workers, n_files)
    # Adjust inner workers if we have spare cores
    if max_outer_workers * max_inner_workers < n_cpu:
        max_inner_workers = min(n_channels, n_cpu // max_outer_workers)

    return max_inner_workers, max_outer_workers


def process_file(file, integration_time, channels, p, m, s, tau_start, output_dir, max_inner_workers):
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
    loader.load_data(max_workers=max_inner_workers)

    # Discard initial tau values below tau_start (they are zero)
    mask = loader.tau > tau_start
    tau = loader.tau[mask]
    g2_norm = loader.g2_norm[:, :, mask]
    countrate = loader.countrate

    # Save results
    base_filename = Path(file).stem
    out_path = output_dir / f"{base_filename}_corr.npz"
    np.savez(out_path, tau=tau, g2_norm=g2_norm, countrate=countrate)
    print(f"Saved correlation data to {out_path}")

    return out_path


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
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_inner_workers, max_outer_workers = choose_worker_counts(len(channels), len(files), reserve_cores=1)
    print(f"Using {max_inner_workers} inner workers and {max_outer_workers} outer workers.")

    futures = []

    with ProcessPoolExecutor(max_workers=max_outer_workers) as executor:
        for file in files:
            futures.append(executor.submit(
                    process_file,
                    file,
                    integration_time,
                    channels,
                    p,
                    m,
                    s,
                    tau_start,
                    output_dir,
                    max_inner_workers
            ))

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Completed processing: {result}")
            except Exception as e:
                print(f"Error processing a file: {e}")
