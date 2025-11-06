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
    t_bin[0] = t_bin[1]  # First bin width is the same as the second bin width
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
    Following [2], the portion of the curve that should be fitted is tau < mua / (10 * musp * k0**2 * Db), where k0
    is the wavenumber of the light in vacuum. For "standard" values (mua = 0.1 1/cm, musp = 10 1/cm, Db = 1e-8 cm^2/s,
    lambda0 = 785 nm), this results in tau < 1.56e-5 s.

    [1] Zhou, C. et al. (2006). "Diffuse optical correlation tomography of cerebral blood flow during cortical
    spreading depression in rat brain".

    [2] Sie, E. et al. (2020). "High-sensitivity multispeckle diffuse correlation spectroscopy".
    """

    def __init__(
            self,
            t_integration: float,
            countrate: float | np.ndarray,
            beta: float,
            n_speckle: int,
            tau_lim: float | np.ndarray = 1.56e-5,
            ensure_decreasing: bool = False
    ):
        """
        Class constructor.

        :param t_integration: Integration time of the measurement [s].
        :param countrate: The count rate of the measurement [Hz]. If a float, the same value is used for all iterations.
            If a vector, its length should be the same as the number of columns in g2_norm.
        :param beta: Light coherence factor.
        :param n_speckle: Number of independent speckles that contribute to the measurement, that is, the number of
            curves that were averaged to obtain g2_norm.
        :param tau_lim: Upper limit of tau for fitting the autocorrelation function [s]. This means that g2_norm is
            fitted for tau < tau_lim to calculate tau_c. If a float, the same value is used for all iterations.
            If a vector, its length should be the same as the number of columns in g2_norm.
        :param ensure_decreasing: A boolean indicating whether to ensure that the standard deviation coming from the
            noise model decreases with tau. If True, the values after the decreasing portion of sigma are set to
            min(sigma).
        """
        self.countrate = countrate
        self.tau_lim = tau_lim
        self.t_integration = t_integration
        self.beta = beta
        self.n_speckle = n_speckle
        self.ensure_decreasing = ensure_decreasing

    def add_noise(self, tau, g2_norm) -> np.ndarray:
        """
        Adds noise to the normalized second-order autocorrelation functions.

        :param tau: Vector of time delays. [s]
        :param g2_norm: 2D array of normalized second-order autocorrelation functions. First dimension is
            iterations, second dimension is time delays.
        :return: The noisy g2_norm. A matrix the same size as g2_norm.
        """
        # Check that the number of columns in g2_norm is the same as the length of tau
        if g2_norm.shape[-1] != len(tau):
            raise ValueError("The number columns in g2_norm should be the same as the length of tau")

        # Check that number of iteration is consistent between g2_norm, countrate, and tau_lim
        if isinstance(self.countrate, np.ndarray):
            if g2_norm.shape[0] != len(self.countrate):
                raise ValueError("g2_norm should have the same number of rows as the length of countrate")
        else:
            self.countrate = np.full(g2_norm.shape[0], self.countrate)
        if isinstance(self.tau_lim, np.ndarray):
            if g2_norm.shape[0] != len(self.tau_lim):
                raise ValueError("g2_norm should have the same number of rows as the length of tau_lim")
        else:
            self.tau_lim = np.full(g2_norm.shape[0], self.tau_lim)

        # Initialize the noisy g2_norm
        g2_norm_noisy = np.empty_like(g2_norm)

        for i in range(g2_norm.shape[0]):
            # Fit the g2 curve to get tau_c
            tau_c = self._fit_tau_c(tau, g2_norm[i, :], self.tau_lim[i], self.beta)

            # Calculate the standard deviation of the normalized g2 curve
            sigma_g2 = sigma_g2_norm(
                tau,
                self.t_integration,
                self.countrate[i],
                self.beta,
                tau_c,
                self.n_speckle
            )

            if self.ensure_decreasing:
                idx_last_good = np.argmin(sigma_g2)
                sigma_g2[idx_last_good + 1:] = sigma_g2[idx_last_good]

            # Add noise to the g2 curve
            g2_norm_noisy[i, :] = np.random.normal(g2_norm[i, :], sigma_g2)

        return g2_norm_noisy

    @staticmethod
    def _fit_tau_c(tau: np.ndarray, g2_norm_single: np.ndarray, tau_lim: float, beta: float) -> float:
        """
        Fits the g2 curve of iteration i to get the corresponding correlation time tau_c, automatically selecting the
        portion of the curve to fit based on the parameters of the medium.
        :param tau: Vector of time delays. [s]
        :param g2_norm_single: 1D normalized second-order autocorrelation function for iteration i.
        :param tau_lim: Upper limit of tau for fitting the autocorrelation function [s].
        :param beta: Light coherence factor.

        :return: The fitted correlation time tau_c [s].
        """
        # Calculate the limit of tau for fitting.
        mask = tau < tau_lim
        tau_fit = tau[mask]
        g2_fit = g2_norm_single[mask]

        def simple_exp(beta, tau, tau_c):
            """
            Simple exponential model for g2.
            """
            return 1 + beta * np.exp(-tau / tau_c)

        def cost_fn(tau_c):
            """
            Cost function for the fit.
            """
            # To help the optimizer, the input tau_c is in hundreds of microseconds. Convert it to seconds.
            g2_simple_exp = simple_exp(beta, tau_fit, tau_c * 1e-4)
            return np.sum((g2_fit - g2_simple_exp) ** 2)

        x0 = np.array(2)  # Initial guess for tau_c in hundreds of microseconds
        bounds = [(0, None)]
        res = opt.minimize(cost_fn, x0, bounds=bounds)

        # Return the fitted tau_c
        return res.x[0] * 1e-4  # Convert tau_c back to seconds


if __name__ == "__main__":
    import fit_dcs.forward.homogeneous_semi_inf as hsi
    from fit_dcs.forward import common
    import matplotlib.pyplot as plt

    tau = np.logspace(-7, -2, 200)
    db_true = 5e-8
    mua = 0.1
    musp = 10
    n = 1.4
    rho = 2.5
    lambda0 = 785
    beta_true = 0.5
    msd_true = common.msd_brownian(tau, db_true)
    g1_norm_true = hsi.g1_norm(msd_true, mua, musp, n, rho, lambda0)
    g2_norm_true = 1 + beta_true * g1_norm_true ** 2

    n_samples = 5
    noise_adder = NoiseAdder(
        t_integration=1,
        countrate=80e3,
        beta=beta_true,
        n_speckle=1,
        ensure_decreasing=True
    )
    g2_norm_noisy = noise_adder.add_noise(g2_norm=np.array([g2_norm_true for _ in range(n_samples)]), tau=tau)
    plt.semilogx(tau, g2_norm_noisy.T, ".")
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel(r"$g_2$")
    plt.show()