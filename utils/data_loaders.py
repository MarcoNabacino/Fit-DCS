import numpy as np
import utils.timetagger
from typing import Dict
import matplotlib.pyplot as plt
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import TimeTagger


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


class DataLoaderALV:
    """
    Data loader for .asc files created by the ALV7004/USB-FAST correlator.

    Reads the data from the files that make up a single measurement and stores it in the class attributes:
    - countrate: Array of shape (n_files, n_channels) containing the countrate for each channel in each file.
    - tau: Array of shape (n_bins,) containing the time delays.
    - g2_norm: Array of shape (n_bins, n_files, n_channels) containing the normalized g2 data.
    - integration_time: Integration time [s].
    """
    N_BINS = 199 # ALV correlator has 199 time bins

    def __init__(self, data_file_paths: list[str], n_channels: int = 4):
        """
        Class constructor.
        :param data_file_paths: List of paths to the data files, including the file extension.
        :param n_channels: Number of channels used in the measurement. Default is 4.
        """
        self.data_file_paths = data_file_paths
        self.n_channels = n_channels

        # Initialize the data arrays
        self.countrate = np.empty((len(self), self.n_channels))
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
            self.countrate[iteration, :] = data["countrate"]
            if iteration == 0:
                self.tau = data["tau"]
                self.integration_time = data["integration_time"]
            self.g2_norm[:, iteration, :] = data["g2_norm"]


class DataLoaderTimeTagger:
    """
    Data loader for the .ttbin files created by the Time Tagger.

    Reads the data from all the files that make up a single measurement, calculates the countrate and the normalized
    autocorrelation according to the specified integration time and stores it in the class attributes:
    - countrate: Array of shape (n_iterations, len(channels)) containing the countrate for each channel in each
        iteration.
    - tau: Array of shape (n_bins,) containing the time delays.
    - g2_norm: Array of shape (n_bins, n_iterations, len(channels)) containing the normalized g2 data.
    """
    T0 = 1e-12 # Time resolution of the Time Tagger [s]

    def __init__(
            self,
            data_file_path: str,
            integration_time: float,
            channels: list[int] = (1, 2, 3, 4),
            n_events: int = 60e4,
            **kwargs
    ):
        """
        Class constructor.
        :param data_file_path: Path to the .ttbin file, including the file extension.
        :param integration_time: Integration time to calculate the autocorrelation [s].
        :param channels: List of channels to be used in the measurement. Default is [1, 2, 3, 4]. Note that they use
            1-based indexing.
        :param n_events: Number of events to be read at a time from the .ttbin file.
        :param kwargs: Additional keyword arguments to be passed to the async_corr function.
        """
        self.data_file_path = data_file_path
        self.integration_time = np.int64(integration_time / self.T0) # Convert integration time to time tagger units
        self.channels = channels
        self.n_events = n_events
        self.correlator_args = kwargs

        # Initialize data arrays.
        self.countrate = np.zeros((len(self), (len(self.channels))))
        p, m, s = self.correlator_args['p'], self.correlator_args['m'], self.correlator_args['s']
        n_bins = (p - p // m) * s
        self.tau = np.zeros(n_bins)
        self.g2_norm = np.empty((n_bins, len(self), len(self.channels)))

    def __len__(self):
        """
        Return the number of iterations in the measurement.
        """
        # If length is not defined, calculate it
        if not hasattr(self, "_len"):
            file_reader = TimeTagger.FileReader(self.data_file_path)
            # Read first photon to set start time.
            buffer = file_reader.getData(1)
            t_start = buffer.getTimestamps()[0]
            t_end = t_start
            # Read all photons to get the total duration of the measurement
            while file_reader.hasData():
                buffer = file_reader.getData(self.n_events)
                t_end = buffer.getTimestamps()[-1]
            self._len = int(np.ceil(float(t_end - t_start) / self.integration_time))

        return self._len

    def load_data(self, plot_interval: int = 0):
        """
        Read the .ttbin file, group photons based on the integration time, calculate the autocorrelation and countrate,
        and store the results in the class attributes.

        :param plot_interval: Interval at which to plot the g2 data. Default is 0, which means no plots.
        """
        file_reader = TimeTagger.FileReader(self.data_file_path)

        # Read first photon to set start time.
        buffer = file_reader.getData(1)
        tags = buffer.getTimestamps()
        channels = buffer.getChannels()
        max_time = tags[-1] # Maximum time tag that has been read

        idx_iteration = 0
        with ProcessPoolExecutor() as executor:
            while len(tags) != 0:
                # Update start time
                start_time = tags[0]
                # Check if we need to read more events to reach the integration time.
                while max_time - start_time < self.integration_time and file_reader.hasData():
                    buffer = file_reader.getData(self.n_events)
                    new_tags = buffer.getTimestamps()
                    new_channels = buffer.getChannels()
                    tags = np.append(tags, new_tags)
                    channels = np.append(channels, new_channels)
                    max_time = tags[-1]
                # Now we either exceeded the integration time or reached the end of the file.

                # Only save in tt the photons before the integration time.
                mask = tags - start_time < self.integration_time
                tags_current, channels_current = tags[mask], channels[mask]
                tags_next, channels_next = tags[~mask], channels[~mask]
                # Create list to store the time tags for each channel.
                tt = [[] for _ in range(len(self.channels))]
                for i_channel, n_channel in enumerate(self.channels):
                    mask_channel = channels_current == n_channel
                    tt[i_channel] = tags_current[mask_channel]

                # Reset tags and channels, filling them with the remaining events
                tags = tags_next
                channels = channels_next

                # Calculate the autocorrelation and the countrate for each channel in parallel, and store the results
                futures = {executor.submit(self._process_channel, i_channel, tt):
                               i_channel for i_channel in range(len(self.channels))}
                for future in concurrent.futures.as_completed(futures):
                    i_channel, g2_norm_out, tau_out, countrate_out = future.result()
                    if idx_iteration == 0:
                        self.tau = tau_out
                    self.g2_norm[:, idx_iteration, i_channel] = g2_norm_out
                    self.countrate[idx_iteration, i_channel] = countrate_out

                # Plot the g2 data if requested
                if plot_interval > 0 and idx_iteration % plot_interval == 0:
                    for i_channel, n_channel in enumerate(self.channels):
                        plt.semilogx(self.tau, self.g2_norm[:, idx_iteration, i_channel],
                                     label=f"Channel {n_channel}")
                    plt.xlabel("Tau [s]")
                    plt.ylabel("g2")
                    plt.ylim(0.8, 1.7)
                    plt.title(f"Iteration {idx_iteration}")
                    plt.legend()
                    plt.show()

                idx_iteration += 1

    def _process_channel(self, i_channel, tt):
        countrate_out = utils.timetagger.countrate(tt[i_channel], self.T0)
        g2_norm_out, tau_out = utils.timetagger.async_corr(np.array(tt[i_channel]), **self.correlator_args)
        return i_channel, g2_norm_out, tau_out, countrate_out


if __name__ == '__main__':
    import cProfile, pstats

    file = "../data/TERm1010.ttbin"
    m = 2
    (p, s) = utils.timetagger.get_correlator_architecture(alpha=7, m=m, tau_max=1e-2, t0=1e-12)
    loader = DataLoaderTimeTagger(
        file,
        integration_time=30e-3,
        channels=[2, 3, 4],
        n_events=int(1e5),
        p=p,
        m=m,
        s=s,
        tau_start=1e-7
    )
    cProfile.run("loader.load_data()", "profile_results")
    p = pstats.Stats("profile_results")
    p.sort_stats("cumulative").print_stats(10)
