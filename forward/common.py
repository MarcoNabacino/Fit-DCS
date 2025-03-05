import numpy as np


def msd_brownian(tau: np.ndarray, db: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for a Brownian motion forward.

    :param tau: Vector of time delays. [s]
    :param db: Diffusion coefficient. [cm^2/s]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return 6 * db * tau


def msd_ballistic(tau: np.ndarray, v_ms: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for a ballistic motion forward.

    :param tau: Vector of time delays. [s]
    :param v_ms: Mean square speed of the scatterers. [cm/s]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return v_ms * tau ** 2


def msd_hybrid(tau: np.ndarray, db: float, v_ms: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for a hybrid (Brownian + ballistic) forward.

    :param tau: Vector of time delays. [s]
    :param db: Diffusion coefficient. [cm^2/s]
    :param v_ms: Mean square speed of the scatterers. [cm/s]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return 6 * db * tau + v_ms * tau ** 2


def effective_reflectance(n: float) -> float:
    """
    Calculates the effective reflectance of a semi-infinite medium, based on a series expansion of the refractive
    index. The formula is taken from [1].

    [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: Theoretical framework,
    recent advances in hardware, analysis, and applications."

    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :return: The effective reflectance of the medium.
    """
    return - 1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n


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
    t_bin = np.diff(tau, prepend=0) # Time bin width
    n = countrate * t_bin # Number of detected photons in each bin
    m = np.arange(1, len(tau) + 1) # Bin index

    prefactor = t_bin / (t_integration * n_speckle)
    a = 1 + beta * np.exp(-tau / (2 * tau_c))
    b = 2 * beta * (1 + np.exp(-tau / tau_c))
    num_c_1 = (1 + np.exp(-t_bin / tau_c)) * (1 + np.exp(-tau / tau_c))
    num_c_2 = 2 * m * (1 - np.exp(-t_bin / tau_c)) * np.exp(-tau / tau_c)
    den_c = 1 - np.exp(-t_bin / tau_c)
    c = beta**2 * (num_c_1 + num_c_2) / den_c

    return 1 / n * np.sqrt(prefactor * (a + b * n + c * n**2))
