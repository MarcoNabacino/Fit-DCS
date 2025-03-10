import numpy as np
import forward.common as common


def g1(msd: np.ndarray, mua: float, musp: float, rho: float, n: float, lambda0: float) -> np.ndarray:
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
    lambda0 *= 1e-7 # Convert to cm
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


def d_factors(msd0: np.ndarray, mua0: float, musp0: float, rho: float, n: float, lambda0: float) -> tuple:
    """
    Calculates the d factors for the DCS Modified Beer-Lambert law for the homogeneous semi-infinite medium. See [1] for
    an explanation.

    [1] Baker, W. et al. (2014), "Modified Beer-Lambert law for blood flow"
    
    :param msd0: Baseline mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua0: Baseline absorption coefficient of the medium. [1/cm]
    :param musp0: Baseline reduced scattering coefficient of the medium. [1/cm]
    :param rho: Source-detector separation. [cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param lambda0: Wavelength of the light source. [nm]
    :return: The d factors. A tuple of three vectors (dr, da, ds), each the same length as tau. dr is the d factor for
        msd, da is the d factor for mua, and ds is the d factor for musp. To get the d factor for the dynamical
        parameter of interest (i.e., Db or v_ms), multiply dr by d(msd)/d(Db) or d(msd)/d(v_ms), respectively.
    """
    lambda0 *= 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0

    k = np.sqrt(3 * musp0 * mua0 + musp0**2 * k0**2 * msd0)
    mu_eff = np.sqrt(3 * musp0 * mua0)
    r = common.effective_reflectance(n)

    z0 = 1 / musp0
    r1 = np.sqrt(rho**2 + z0**2)
    zb = 2 / 3 * (1 + r) / (1 - r) * z0
    rb = np.sqrt((2 * zb + z0)**2 + rho**2)

    # dr
    num = np.exp(-k * r1) - np.exp(-k * rb)
    den = np.exp(-k * r1) / r1 - np.exp(-k * rb) / rb
    dr = musp0**2 * k0**2 / k * num / den

    # da
    num0 = np.exp(-mu_eff * r1) - np.exp(-mu_eff * rb)
    den0 = np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * rb) / rb
    da = 3 * musp0 * (num / (k * den) - num0 / (mu_eff * den0))

    # ds
    # Derivatives with respect to musp
    dk = (3 * mua0 + 2 * k0**2 * musp0 * msd0) / (2 * k)
    dmu_eff = 3 * mua0 / (2 * mu_eff)
    dr1 = -z0**3 / r1
    drb = -z0 / rb * (2 * zb + z0)**2
    # Factors with derivatives
    f1 = r1 * dk + (k + 1 / r1) * dr1
    f10 = r1 * dmu_eff + (mu_eff + 1 / r1) * dr1
    fb = rb * dk + (k + 1 / rb) * drb
    fb0 = rb * dmu_eff + (mu_eff + 1 / rb) * drb
    # Numerators
    num = f1 * np.exp(-k * r1) / r1 - fb * np.exp(-k * rb) / rb
    num0 = f10 * np.exp(-mu_eff * r1) / r1 - fb0 * np.exp(-mu_eff * rb) / rb
    # Denominators
    den = np.exp(-k * r1) / r1 - np.exp(-k * rb) / rb
    den0 = np.exp(-mu_eff * r1) / r1 - np.exp(-mu_eff * rb) / rb
    ds = 2 * (num / den - num0 / den0)

    return dr, da, ds