import numpy as np
from typing import Dict
import TimeTagger


class DataLoaderALV:
    """
    Data loader for .asc files created by the ALV7004/USB-FAST correlator.

    Reads the data from the files that make up a single measurement and stores it in the class attributes:
    - countrate: Array of shape (n_channels, n_files) containing the countrate for each channel in each file.
    - tau: Array of shape (n_bins,) containing the time delays.
    - g2_norm: Array of shape (n_bins, n_files, n_channels) containing the normalized g2 data.
    - integration_time: Integration time [s].
    """
    N_BINS = 199 # ALV correlator has 199 time bins

    def __init__(self, data_file_paths: list[str], n_channels: int = 4):
        """
        Class constructor.
        :param data_file_paths: List of paths to the data files.
        :param n_channels: Number of channels used in the measurement. Default is 4.
        """
        self.data_file_paths = data_file_paths
        self.n_channels = n_channels

        # Initialize the data arrays
        self.countrate = np.empty((self.n_channels, len(self)))
        self.tau = np.empty(self.N_BINS)
        self.g2_norm = np.empty((self.N_BINS, len(self), self.n_channels))
        self.integration_time = np.nan # Integration time [s]

    def __len__(self):
        """
        Return the number of data files.
        """
        return len(self.data_file_paths)

    def load_data(self):
        """
        Load the data from the .asc files and store it in the class attributes.
        """
        for iteration in range(len(self)):
            filename = self.data_file_paths[iteration]
            data = read_asc(filename, n_ch=self.n_channels, n_bins=self.N_BINS)
            self.countrate[:, iteration] = data["countrate"]
            if iteration == 0:
                self.tau = data["tau"]
                self.integration_time = data["integration_time"]
            self.g2_norm[:, iteration, :] = data["g2_norm"]


def read_asc(filename, n_ch: int = 4, n_bins: int = 199) -> Dict:
    """
    Read the data from a single .asc file.
    :param filename: Path to the .asc file.
    :param n_ch: Number of channels used in the measurement. Default is 4.
    :param n_bins: Number of bins in the g2 data. Default is 199.
    :return: Dictionary containing the data. Keys are "date", "time", "integration_time", "countrate", "tau", "g2_norm".
    """
    with open(filename, "r") as file:
        # Keep reading lines until you find the line that starts with "Date", which contains the date between quotes
        line = file.readline()
        while "Date" not in line:
            line = file.readline()
        date = line.split('"')[1]
        # Keep reading lines until you find the line that starts with "Time", which contains the time between quotes
        while "Time" not in line:
            line = file.readline()
        time = line.split('"')[1]
        # Keep reading lines until you find the line that starts with "Duration", which contains the integration time
        # at the end of the line
        while "Duration" not in line:
            line = file.readline()
        integration_time = float(line.split()[-1])
        # Keep reading lines until you find the line that starts with "MeanCR0", which contains the
        # countrate for channel 0, while the next lines contain the countrate for the other channels
        countrate = np.empty(n_ch)
        while "MeanCR0" not in line:
            line = file.readline()
        countrate[0] = float(line.split()[-1])
        for i in range(1, n_ch):
            line = file.readline()
            countrate[i] = float(line.split()[-1])
        # Keep reading lines until you find the line that contains "Correlation" in quotes, after which the tau and g2
        # data starts.
        while "Correlation" not in line:
            line = file.readline()
        # The next lines contain tau and the g2 for each channel, until you find an empty line. Note that the
        # file might have fewer lines than the expected n_bins, typically for the last iteration.
        tau = np.empty(n_bins)
        g2_norm = np.empty((n_bins, n_ch))
        for i_bin in range(n_bins):
            line = file.readline()
            if not line:
                break
            values = line.encode("utf-8").split()
            tau[i_bin] = float(values[0])
            for ch in range(n_ch):
                g2_norm[i_bin, ch] = float(values[ch + 1])

    # The correlator saves g2 - 1, so we need to add 1 to get the normalized g2
    g2_norm += 1
    # Tau is stored in ms, so we need to convert it to s
    tau *= 1e-3

    return dict(
        date=date,
        time=time,
        integration_time=integration_time,
        countrate=countrate,
        tau=tau,
        g2_norm=g2_norm
    )


def async_corr(t: np.ndarray, p: int, m: int, s: int, t0: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the autocorrelation function using the asynchronous algorithm described in [1].

    [1] Wahl, M. et al. (2003), "Fast calculation of fluorescence correlation data with asynchronous time-correlated
    single photon counting".

    :param t: Vector of photon time tags (unit specified by t0).
    :param p: Number of time bins in each linear correlator.
    :param m: Binning ratio. Must divide p.
    :param s: Number of linear correlators.
    :param t0: Time resolution of the time tagger [s]. Default is 1 ps.
    :return: A tuple containing the normalized autocorrelation function g2_norm and the corresponding lag times tau.
    """
    # Check that m divides p
    if p % m != 0:
        raise ValueError("The binning ratio m must divide the number of time bins per linear correlator p.")

    n_bins = (p - p // m) * s
    n_tags = len(t)
    weights = np.ones(n_tags) # Initialize photon weights to 1
    dt = t.max() - t.min() # Measurement duration

    delta = 1 # Increment of lag time in the linear correlator
    shift = 0 # Initial lag time
    tau_index = 0
    autocorr = np.zeros(n_bins)
    autotime = np.zeros(n_bins)

    for _ in range(s):
        # Compute the weights for the current linear correlator by eliminating duplicate tags and summing the weights.
        t, indices = np.unique(t, return_inverse=True)
        weights = np.bincount(indices, weights=weights).astype(int)

        for _ in range(int(p / m), p):
            shift += delta # Increment the lag time, this is p * delta
            lag = shift // delta
            tp = t + lag

            n = 0
            r = 0
            while n < len(t) - 1 and r < len(tp) - 1:
                if t[n] < tp[r]:
                    n += 1
                elif tp[r] < t[n]:
                    r += 1
                elif t[n] == tp[r]:
                    autocorr[tau_index] += weights[n] * weights[r]
                    n += 1
                    r += 1

            autocorr[tau_index] /= delta
            autotime[tau_index] = shift
            tau_index += 1
        delta *= m # Increase bin width, this is m^s
        t = t // m

    g2_norm = autocorr * dt**2 / (n_tags**2 * (dt - autotime))
    tau = autotime * t0

    return g2_norm, tau