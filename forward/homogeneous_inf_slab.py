import numpy as np
import forward.common as common


def g1_transmittance(msd: np.ndarray, mua: float, musp: float, rho: float, n: float, lambda0: float,
                     d: float, m_max: int) -> np.ndarray:
    """
    Calculates the unnormalized first-order autocorrelation function in transmittance for a laterally infinite slab.
    The prefactor, which depends on the source term, is set to 1 instead. Since only the normalized g1 is used, this is
    not a problem.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Scattering coefficient of the medium. [1/cm]
    :param rho: Lateral distance from the source. [cm]
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param d: Thickness of the slab. [cm]
    :param m_max: Maximum index of the truncated summation. The summation goes from -m_max to +m_max, inclusive, so
        the total number of terms is 2 * m_max + 1.
    :return: The unnormalized first-order autocorrelation function in transmittance. A vector the same length as tau.
    """
    lambda0 *= 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp
    r = common.effective_reflectance(n)
    zb = 2 / (3 * musp) * (1 + r) / (1 - r)
    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    g1 = 0
    for m in range(-m_max, m_max + 1):
        z1 = d * (1 - 2 * m) - 4 * m * zb - z0
        z2 = d * (1 - 2 * m) - (4 * m  - 2) * zb + z0
        r1 = np.sqrt(rho**2 + z1**2)
        r2 = np.sqrt(rho**2 + z2**2)
        term1 = (rho**2 + z1**2) ** (-1.5) * z1 * np.exp(-k * r1) * (1 + k * r1)
        term2 = (rho**2 + z2**2) ** (-1.5) * z2 * np.exp(-k * r2) * (1 + k * r2)
        g1 += term1 - term2

    return g1


def g1_transmittance_norm(msd: np.ndarray, mua: float, musp: float, rho: float, n: float, lambda0: float,
                          d: float, m_max: int) -> np.ndarray:
    """
    Calculates the normalized first-order autocorrelation function in transmittance for a laterally infinite slab.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Scattering coefficient of the medium. [1/cm]
    :param rho: Lateral distance from the source. [cm]
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param d: Thickness of the slab. [cm]
    :param m_max: Maximum index of the truncated summation. The summation goes from -m_max to +m_max, inclusive, so
        the total number of terms is 2 * m_max + 1.
    :return: The normalized first-order autocorrelation function in transmittance. A vector the same length as tau.
    """
    return (g1_transmittance(msd, mua, musp, rho, n, lambda0, d, m_max) /
            g1_transmittance(0, mua, musp, rho, n, lambda0, d, m_max))