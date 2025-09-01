import numpy as np


def msd_brownian(tau: np.ndarray, db: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for Brownian motion.

    :param tau: Vector of time delays. [s]
    :param db: Diffusion coefficient. [cm^2/s]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return 6 * db * tau


def d_msd_brownian(tau: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative of the mean-square displacement with respect to Db for Brownian motion.

    :param tau: Vector of time delays. [s]
    :return: The derivative of the mean-square displacement with respect to Db. A vector the same length as tau.
    """
    return 6 * tau


def msd_ballistic(tau: np.ndarray, v_ms: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for ballistic motion.

    :param tau: Vector of time delays. [s]
    :param v_ms: Mean square speed of the scatterers. [cm^2/s^2]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return v_ms * tau ** 2


def d_msd_ballistic(tau: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative of the mean-square displacement with respect to v_ms for ballistic motion.

    :param tau: Vector of time delays. [s]
    :return: The derivative of the mean-square displacement with respect to v_ms. A vector the same length as tau.
    """
    return tau ** 2


def msd_hybrid(tau: np.ndarray, db: float, v_ms: float) -> np.ndarray:
    """
    Calculates the mean-square displacement for a hybrid (Brownian + ballistic) model.

    :param tau: Vector of time delays. [s]
    :param db: Diffusion coefficient. [cm^2/s]
    :param v_ms: Mean square speed of the scatterers. [cm^2/s^2]
    :return: The mean-square displacement. A vector the same length as tau. [cm^2]
    """
    return 6 * db * tau + v_ms * tau ** 2


def a_coefficient_boundary(n: float) -> float:
    """
    Calculates the A coefficient for the boundary condition.

    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :return: The A coefficient for the boundary condition.
    """
    r = effective_reflectance(n)
    return (1 + r) / (1 - r)


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
    return -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
