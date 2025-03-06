import numpy as np
from typing import Dict

class DataLoaderALV:
    """
    Data loader for .asc files created by the ALV7004/USB-FAST correlator.

    Reads the data from the files that make up a single measurement and stores it in the class attributes.
    """
    def __init__(self, data_file_paths: list[str]):
        """
        Class constructor.
        :param data_file_paths: List of paths to the data files.
        """
        self.data_file_paths = data_file_paths

        # Initialize the data arrays
        self.n_bins = 199 # ALV correlator has 199 time bins
        self.n_ch = 4 # Assume that all 4 channels were used
        self.countrate = np.empty((self.n_ch, len(self)))
        self.tau = np.empty(self.n_bins)
        self.g2_norm = np.empty((self.n_bins, len(self), self.n_ch))
        self.integration_time = np.nan # Integration time [s]

    def __len__(self):
        """
        Return the number of data files.
        """
        return len(self.data_file_paths)

    def load_data(self):
        """
        Load the data from the .asc files and store it in the class attributes.
        Assume that all 4 channels were used.
        """
        for iteration in range(len(self)):
            filename = self.data_file_paths[iteration]
            data = read_asc(filename)
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