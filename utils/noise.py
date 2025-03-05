import numpy as np
import scipy.optimize as opt


def sigma_g2_norm(tau: np.ndarray, t_integration: float, countrate: float, beta: float, tau_c: float,
                  n_speckle: int) -> np.ndarray:
    """
    Calculates the standard deviation of the normalized second-order autocorrelation function g2_norm using the DCS
    noise model [1], with the extension to the multispeckle case [2]. Notation follows [2], in particular Eq. (5).

    [1] Zhou, C. et al. (2006). "Diffuse optical correlation tomography of cerebral blood flow during cortical
    spreading depression in rat brain".

    [2] Sie, E. et al. (2020). "High-sensitivity multispeckle diffuse correlation spectroscopy".

    :param tau: Vector of time delays. [s]
    :param t_integration: Integration time of the measurement. [s]
    :param countrate: Detected count rate of the measurement. [Hz]
    :param beta: Light coherence factor.
    :param tau_c: The correlation time for a simple exponential decay, i.e.,
        g2(tau) = 1 + beta * exp(-tau/t_correlation).
    :param n_speckle: The number of independent speckles contributing to the measurement.
    :return: The standard deviation of the normalized second-order autocorrelation function g2_norm. A vector the same
        length as tau.
    """
    t_bin = np.diff(tau, prepend=0)  # Time bin width
    n = countrate * t_bin  # Number of detected photons in each bin
    m = tau / t_bin  # Delay time bin index


    prefactor = t_bin / (t_integration * n_speckle)
    a = 1 + beta * np.exp(-tau / (2 * tau_c))
    b = 2 * beta * (1 + np.exp(-tau / tau_c))
    num_c_1 = (1 + np.exp(-t_bin / tau_c)) * (1 + np.exp(-tau / tau_c))
    num_c_2 = 2 * m * (1 - np.exp(-t_bin / tau_c)) * np.exp(-tau / tau_c)
    den_c = 1 - np.exp(-t_bin / tau_c)
    c = beta ** 2 * (num_c_1 + num_c_2) / den_c

    return 1 / n * np.sqrt(prefactor * (a + b * n + c * n ** 2))


class NoiseAdder:
    """
    Class for adding noise to DCS data.

    It takes as input a matrix of normalized second-order autocorrelation functions, representing a DCS measurement,
    and adds noise to it based on specified experimental parameters (e.g., integration time, count rate, etc.).

    The noise model is based on [1], with the extension to the multispeckle case [2].

    To get the correlation time tau_c, a fit of the g2 curve with a simple exponential model
    g2_simple_exp = 1 + beta * np.exp(-tau / tau_c) is performed. Note that this model is a good approximation for
    Brownian motion, or hybrid motion with a dominant Brownian component, but not for ballistic motion.
    Following [2], the portion of the curve that is fitted is tau < mua / (10 * musp * k0**2 * Db), where k0 is the
    wavenumber of the light in vacuum. For the default values (mua = 0.1 1/cm, musp = 10 1/cm, Db = 1e-8 cm^2/s,
    lambda0 = 785 nm), this results in tau < 1.56e-5 s.

    [1] Zhou, C. et al. (2006). "Diffuse optical correlation tomography of cerebral blood flow during cortical
    spreading depression in rat brain".

    [2] Sie, E. et al. (2020). "High-sensitivity multispeckle diffuse correlation spectroscopy".
    """

    def __init__(
            self,
            g2_norm: np.ndarray,
            tau: np.ndarray,
            t_integration: float,
            countrate: float | np.ndarray,
            beta: float,
            n_speckle: int,
            mua: float | np.ndarray = 0.1,
            musp: float | np.ndarray = 10,
            db: float | np.ndarray = 1e-8,
            lambda0: float = 785,
    ):
        """
        Class constructor.

        :param g2_norm: Matrix of normalized second-order autocorrelation functions. Each column corresponds to a
            different iteration, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param tau: Vector of time delays [s].
        :param t_integration: Integration time of the measurement [s].
        :param countrate: The count rate of the measurement [Hz]. If a float, the same value is used for all iterations.
            If a vector, its length should be the same as the number of columns in g2_norm.
        :param beta: Light coherence factor.
        :param n_speckle: Number of independent speckles that contribute to the measurement, that is, the number of
            curves that were averaged to obtain g2_norm.
        :param mua: Absorption coefficient of the medium [1/cm]. If a float, the same value is used for all iterations.
            If a vector, a different value is used for each iteration, and the length of the vector should be the same
            as the number of columns in g2_norm. Default is 0.1 1/cm.
        :param musp: Reduced scattering coefficient of the medium [1/cm]. If a float, the same value is used for all
            iterations. If a vector, a different value is used for each iteration, and the length of the vector should
            be the same as the number of columns in g2_norm. Default is 10 1/cm.
        :param db: Brownian motion diffusion coefficient [cm^2/s]. If a float, the same value is used for all
            iterations. If a vector, a different value is used for each iteration, and the length of the vector should
            be the same as the number of columns in g2_norm. Default is 1e-8 cm^2/s.
        :param lambda0: Wavelength of the light source [nm]. Default is 785 nm.
        """
        # Check that the number of rows in g2_norm is the same as the length of tau
        if g2_norm.shape[0] == len(tau):
            self.tau = tau
            self.g2_norm = g2_norm
        else:
            raise ValueError("The number of rows in g2_norm should be the same as the length of tau")

        # Check that countrate is either a float or an array of the same length as the number of columns in g2_norm
        if isinstance(countrate, (float, int)):
            self.countrate = np.full(len(self), countrate)
        elif isinstance(countrate, np.ndarray):
            if len(countrate) == len(self):
                self.countrate = countrate
            else:
                raise ValueError(
                    "countrate should be a float or an array of the same length as the number of columns in "
                    "g2_norm")
        else:
            raise ValueError("countrate should be a float or an array")

        # Check that mua, musp, and db are either floats or arrays of the same length as the number of columns in
        # g2_norm
        if isinstance(mua, (float, int)):
            self.mua = np.full(len(self), mua)
        elif isinstance(mua, np.ndarray):
            if len(mua) == len(self):
                self.mua = mua
            else:
                raise ValueError("mua should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")
        else:
            raise ValueError("mua should be a float or an array")
        if isinstance(musp, (float, int)):
            self.musp = np.full(len(self), musp)
        elif isinstance(musp, np.ndarray):
            if len(musp) == len(self):
                self.musp = musp
            else:
                raise ValueError("musp should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")
        else:
            raise ValueError("musp should be a float or an array")
        if isinstance(db, (float, int)):
            self.db = np.full(len(self), db)
        elif isinstance(db, np.ndarray):
            if len(db) == len(self):
                self.db = db
            else:
                raise ValueError("db should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")
        else:
            raise ValueError("db should be a float or an array")

        self.t_integration = t_integration
        self.beta = beta
        self.n_speckle = n_speckle
        self.lambda0 = lambda0

    def __len__(self):
        """
        Returns the number of iterations.
        """
        return self.g2_norm.shape[1]

    def add_noise(self) -> np.ndarray:
        """
        Adds noise to the normalized second-order autocorrelation functions.

        :return: The noisy g2_norm. A matrix the same size as g2_norm.
        """
        # Initialize the noisy g2_norm
        g2_norm_noisy = np.zeros_like(self.g2_norm)

        for i in range(len(self)):
            # Fit the g2 curve to get tau_c
            tau_c = self._fit_tau_c(i)

            # Calculate the standard deviation of the normalized g2 curve
            sigma_g2 = sigma_g2_norm(
                self.tau,
                self.t_integration,
                self.countrate[i],
                self.beta,
                tau_c,
                self.n_speckle
            )

            # Add noise to the g2 curve
            g2_norm_noisy[:, i] = self.g2_norm[:, i] + np.random.normal(0, sigma_g2)

        return g2_norm_noisy

    def _fit_tau_c(self, i: int) -> float:
        """
        Fits the g2 curve of iteration i to get the corresponding correlation time tau_c, automatically selecting the
        portion of the curve to fit based on the parameters of the medium.
        :param i: Index of the iteration.
        :return: The fitted correlation time tau_c [s].
        """
        # Calculate the limit of tau for fitting.
        k0 = 2 * np.pi / (self.lambda0 * 1e-7)  # Convert lambda0 to cm
        tau_lim = self.mua[i] / (10 * self.musp[i] * k0 ** 2 * self.db[i])
        tau_fit = self.tau[self.tau < tau_lim]
        g2_fit = self.g2_norm[self.tau < tau_lim, i]

        return fit_tau_c(self.beta, tau_fit, g2_fit)


def fit_tau_c(beta: float, tau_fit: np.ndarray, g2_fit: np.ndarray) -> float:
    """
    Fits the correlation time tau_c of a g2 curve to a simple exponential model g2(tau) = 1 + beta * exp(-tau/tau_c).
    :param beta: Light coherence factor.
    :param tau_fit: Vector of time delays for fitting. [s]
    :param g2_fit: Vector of normalized second-order autocorrelation function values for fitting.
    :return: The fitted correlation time tau_c [s].
    """
    # Define the model function for the fit
    def simple_exp(beta, tau, tau_c):
        return 1 + beta * np.exp(-tau / tau_c)

    # Define the cost function for the fit
    def cost(tau_c):
        # To help the optimizer, the input tau_c is in hundreds of microseconds. Convert it to seconds.
        g2_simple_exp = simple_exp(beta, tau_fit, tau_c * 1e-4)
        return np.sum((g2_fit - g2_simple_exp) ** 2)

    # Perform the fit
    x0 = np.array(2) # Initial guess for tau_c in hundreds of microseconds
    res = opt.minimize(cost, x0)

    # Return the fitted tau_c
    return res.x[0] * 1e-4 # Convert tau_c back to seconds

