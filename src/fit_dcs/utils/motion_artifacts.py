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


import numpy as np
from typing import Literal, Callable


def remove_cycling_artifacts(
        db_raw: np.ndarray,
        cycling_frequency: np.ndarray,
        indices_baseline: np.ndarray,
        indices_onset: np.ndarray,
        averaging_method: Literal["mean", "median"] | Callable = "mean",
) -> tuple[np.ndarray, float]:
    """
    Removes motion artifacts due to cycling from the Brownian diffusion coefficient, using the algorithm described in
    [1].

    [1] Quaresima, V. et al. (2019). "Diffuse correlation spectroscopy and frequency-domain near-infrared spectroscopy
    for measuring microvascular blood flow in dynamically exercising human muscles".

    :param db_raw: Brownian diffusion coefficient with motion artefacts.
    :param cycling_frequency: Cycling frequency for each time point. Must be the same length as db_raw.
    :param indices_baseline: Indices of time points before cycling starts over which to calculate the baseline.
    :param indices_onset: Indices of time points after cycling starts over which to calculate the value of db that is
        set to the baseline.
    :param averaging_method: Method to use for averaging the baseline and cycling values.
        Can be "mean", "median", or a callable function that takes a 1D array and returns a scalar. Default is "mean".
    :return: A tuple containing:
        - The Brownian diffusion coefficient with motion artefacts removed. Same shape and unit as db_raw.
        - The value of the k coefficient (see [1] for details). Unit: [db_raw unit / cycling_frequency unit].
    """

    if len(db_raw) != len(cycling_frequency):
        raise ValueError(f"db_raw and cycling_frequency must be the same length. "
                         f"Got {len(db_raw)} and {len(cycling_frequency)}.")

    if averaging_method == "mean":
        avg_func = np.mean
    elif averaging_method == "median":
        avg_func = np.median
    elif callable(averaging_method):
        avg_func = averaging_method
    else:
        raise ValueError(f"averaging_method must be 'mean', 'median', or a callable function. Got {averaging_method}.")

    db_baseline = avg_func(db_raw[indices_baseline])
    db_onset = avg_func(db_raw[indices_onset])
    freq_onset = avg_func(cycling_frequency[indices_onset])

    k = 6 * (db_onset - db_baseline) / freq_onset
    db_corrected = db_raw - k / 6 * cycling_frequency

    return db_corrected, k
