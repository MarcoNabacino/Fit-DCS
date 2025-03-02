import numpy as np
import forward.common as common


def g1(msd: np.ndarray | float, mua: float, musp: float, rho: float, n: float, lambda0: float) -> np.ndarray:
    """
    Calculates the unnormalized first-order autocorrelation function g1 for a homogeneous semi-infinite medium.

    The prefactor, which depends on the source term, is set to 1 instead. Since only the normalized g1 is used, this is
    not a problem.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Reduced scattering coefficient of the medium. [1/cm]
    :param rho: Source-detector separation. [cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param lambda0: Wavelength of the light source. [nm]
    :return: The unnormalized first-order autocorrelation function g1. A vector the same length as tau.
    """
    lambda0 = lambda0 * 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp
    r = common.effective_reflectance(n)
    zb = 2 / (3 * musp) * (1 + r) / (1 - r)
    r1 = np.sqrt(rho**2 + z0**2)
    r2 = np.sqrt(rho**2 + (z0 + 2 * zb)**2)

    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    term1 = np.exp(-k * r1) / r1
    term2 = np.exp(-k * r2) / r2
    return term1 - term2


def g1_norm(msd: np.ndarray | float, mua: float, musp: float, rho: float, n: float, lambda0: float) -> np.ndarray:
    """
    Calculates the normalized first-order autocorrelation function g1 for a homogeneous semi-infinite medium.
    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Reduced scattering coefficient of the medium. [1/cm]
    :param rho: Source-detector separation. [cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param lambda0: Wavelength of the light source. [nm]
    :return: The normalized first-order autocorrelation function g1. A vector the same length as tau.
    """
    return g1(msd, mua, musp, rho, n, lambda0) / g1(0, mua, musp, rho, n, lambda0)
