import numpy as np
from core.libloader import get_lib_path
import ctypes
from ctypes import POINTER, c_int64, c_double


_async_corr_lib = ctypes.CDLL(get_lib_path("libasync_corr"))  # Load the C library for async correlation

_async_corr_lib.async_corr.argtypes = [
    POINTER(c_int64), # t
    c_int64,          # n_tags
    c_int64,          # p
    c_int64,          # m
    c_int64,          # s
    c_double,         # tau_start
    c_double,         # t0
    POINTER(c_double), # g2_out
    POINTER(c_double), # tau_out
]
_async_corr_lib.async_corr.restype = None


def async_corr_c(t: np.ndarray, p: int, m: int, s: int, tau_start: float = 20e-9, t0: float = 1e-12) \
    -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the autocorrelation function using the asynchronous algorithm described in [1] (C implementation).

    [1] Wahl, M. et al. (2003), "Fast calculation of fluorescence correlation data with asynchronous time-correlated
    single photon counting".

    :param t: Vector of photon time tags (unit specified by t0).
    :param p: Number of time bins in each linear correlator.
    :param m: Binning ratio. Must divide p.
    :param s: Number of linear correlators.
    :param tau_start: First value of the lag time [s] for which to calculate the autocorrelation. Default is 20 ns.
    :param t0: Time resolution of the time tagger [s]. Default is 1 ps.
    :return: A tuple containing the normalized autocorrelation function g2_norm and the corresponding lag times tau.
    """
    t = np.require(t, dtype=np.int64, requirements='C')
    n_tags = t.shape[0]
    n_bins = (p - (p // m)) * s

    g2_out = np.zeros(n_bins, dtype=np.float64)
    tau_out = np.zeros(n_bins, dtype=np.float64)

    _async_corr_lib.async_corr(
        t.ctypes.data_as(POINTER(c_int64)),
        c_int64(n_tags),
        c_int64(p),
        c_int64(m),
        c_int64(s),
        c_double(tau_start),
        c_double(t0),
        g2_out.ctypes.data_as(POINTER(c_double)),
        tau_out.ctypes.data_as(POINTER(c_double))
    )

    return g2_out, tau_out


def get_correlator_architecture(alpha: int, m: int, tau_max: float, t0: float) -> tuple[int, int]:
    """
    Calculate the correlator architecture parameters p and s given the parameters alpha, m and tau_max. Info in [1].

    [1] Magatti, D. and Ferri, F. (2001), "Fast multi-tau real-time software correlator for dynamic light scattering".

    :param alpha: Ratio of tau to bin width. For an accuracy better than 0.1%, alpha should be at least 7 [1].
    :param m: Binning ratio. This is typically 2 in hardware correlators.
    :param tau_max: Maximum lag time of the correlation function [s].
    :param t0: Time resolution of the time tagger [s].
    :return: A tuple containing the correlator parameters p and s.
    """
    p = alpha * m
    s = np.ceil(np.log(tau_max / t0 / alpha) / np.log(m)).astype(int)
    return p, s


def countrate(t: np.ndarray, t0: float = 1e-12) -> np.float64:
    """
    Calculate the count rate of a time tag vector.

    :param t: Vector of photon time tags.
    :param t0: Time resolution of the time tagger [s]. Default is 1 ps.
    :return: The count rate in kHz.
    """
    return len(t) / ((t[-1] - t[0]) * t0) / 1000


def async_corr(t: np.ndarray, p: int, m: int, s: int, tau_start: float = 20e-9, t0: float = 1e-12) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the autocorrelation function using the asynchronous algorithm described in [1].

    [1] Wahl, M. et al. (2003), "Fast calculation of fluorescence correlation data with asynchronous time-correlated
    single photon counting".

    :param t: Vector of photon time tags (unit specified by t0).
    :param p: Number of time bins in each linear correlator.
    :param m: Binning ratio. Must divide p.
    :param s: Number of linear correlators.
    :param tau_start: First value of the lag time [s] for which to calculate the autocorrelation. Default is 20 ns.
    :param t0: Time resolution of the time tagger [s]. Default is 1 ps.
    :return: A tuple containing the normalized autocorrelation function g2_norm and the corresponding lag times tau.
    """
    # Check that m divides p
    if p % m != 0:
        raise ValueError("The binning ratio m must divide the number of time bins per linear correlator p.")

    n_overlapped_bins = p // m # Precomputed for speed
    n_bins = (p - n_overlapped_bins) * s
    n_tags = len(t)
    weights = np.ones_like(t, dtype=np.int64) # Initialize photon weights to 1
    dt = np.float64(t[-1] - t[0]) # Measurement duration

    delta = np.int64(1) # Increment of lag time in the linear correlator
    shift = np.int64(0) # Initial lag time
    shift_start = np.int64(tau_start / t0) # Convert tau_start to time tagger units
    tau_index = np.int64(0)
    autocorr = np.zeros(n_bins, dtype=np.int64)
    autotime = np.zeros(n_bins, dtype=np.int64)
    bin_width = np.zeros(n_bins, dtype=np.int64)

    for _ in range(s):
        # If t has duplicates, eliminate them and adjust the weights
        if np.any(np.diff(t) == 0):
            # Compute the weights for the current linear correlator by eliminating duplicate tags and summing the
            # weights.
            t, indices = unique_with_inverse(t)
            weights = np.bincount(indices, weights).astype(np.int64)

        for _ in range(n_overlapped_bins, p):
            shift += delta # Increment the lag time, this is p * delta
            lag = shift // delta
            # Only compute the autocorrelation for lag times greater than tau_start
            if shift < shift_start:
                autotime[tau_index] = shift
                bin_width[tau_index] = delta
                tau_index += 1
                continue

            tp = t + lag
            # Use binary search for fast lookup
            matches = np.searchsorted(t, tp)
            # Ensure valid matches
            valid = matches < len(t)
            valid[valid] &= t[matches[valid]] == tp[valid]

            # Update autocorrelation
            autocorr[tau_index] += np.sum(weights[valid] * weights[matches[valid]])
            # Store lag time
            autotime[tau_index] = shift
            # Store bin width
            bin_width[tau_index] = delta

            tau_index += 1

        delta *= m # Increase bin width, this is m^s
        t //= m # Coarsen resolution

    # Compute final normalized autocorrelation
    norm_factor = dt**2 / n_tags**2
    g2_norm = norm_factor * autocorr.astype(np.float64) / bin_width.astype(np.float64) / (dt - autotime)
    tau = autotime * t0

    return g2_norm, tau


def unique_with_inverse(sorted_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unique values and inverse indices for a sorted array.
    """
    mask = np.empty(len(sorted_arr), dtype=bool)
    mask[0] = True
    mask[1:] = sorted_arr[1:] != sorted_arr[:-1]  # Identify unique elements

    unique_vals = sorted_arr[mask]  # Extract unique values
    inverse_indices = np.cumsum(mask) - 1  # Compute inverse mapping

    return unique_vals, inverse_indices


if __name__ == "__main__":
    import TimeTagger
    import time
    import matplotlib.pyplot as plt

    reader = TimeTagger.FileReader("../examples/data/TERm1010.ttbin")
    buffer = reader.getData(1e6)
    tags = buffer.getTimestamps()
    channels = buffer.getChannels()

    tt = tags[channels == 1]
    print(f"Number of tags: {len(tt)}")
    cr = countrate(tt)
    print(f"Count rate: {cr/1000.0:.3f} kHz")

    p, s = get_correlator_architecture(alpha=7, m=2, tau_max=1e-2, t0=1e-12)
    tau_start = 1e-7
    start_time = time.time()
    g2_norm, tau = async_corr_c(tt, p, m=2, s=s, tau_start=tau_start)
    print(f"Async correlation took {time.time() - start_time:.3f} seconds")

    plt.semilogx(tau, g2_norm)
    plt.xlabel("Lag time (s)")
    plt.ylabel("Normalized autocorrelation g2")
    plt.title("Asynchronous Correlation Function")
    plt.grid()
    plt.show()
