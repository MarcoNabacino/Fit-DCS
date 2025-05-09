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
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp)
    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    g1 = 0
    for m in range(-m_max, m_max + 1):
        z1 = d * (1 - 2 * m) - 4 * m * zb - z0
        z2 = d * (1 - 2 * m) - (4 * m  - 2) * zb + z0
        r1 = np.sqrt(rho**2 + z1**2)
        r2 = np.sqrt(rho**2 + z2**2)
        term1 = z1 / r1**3 * (1 + k * r1) * np.exp(-k * r1)
        term2 = z2 / r2**3 * (1 + k * r2) * np.exp(-k * r2)
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


def g1_reflectance(msd: np.ndarray, mua: float, musp: float, rho: float, n: float, lambda0: float,
                     d: float, m_max: int) -> np.ndarray:
    """
    Calculates the unnormalized first-order autocorrelation function in reflectance for a laterally infinite slab.
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
    :return: The unnormalized first-order autocorrelation function in reflectance. A vector the same length as tau.
    """
    lambda0 *= 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp)
    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    g1 = 0
    for m in range(-m_max, m_max + 1):
        z3 = -2 * m * d - 4 * m * zb - z0
        z4 = -2 * m * d - (4 * m  - 2) * zb + z0
        r3 = np.sqrt(rho**2 + z3**2)
        r4 = np.sqrt(rho**2 + z4**2)
        term1 = z3 / r3**3 * (1 + k * r3) * np.exp(-k * r3)
        term2 = z4 / r4**3 * (1 + k * r4) * np.exp(-k * r4)
        g1 += term1 - term2

    return g1


def g1_reflectance_norm(msd: np.ndarray, mua: float, musp: float, rho: float, n: float, lambda0: float,
                          d: float, m_max: int) -> np.ndarray:
    """
    Calculates the normalized first-order autocorrelation function in reflectance for a laterally infinite slab.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Scattering coefficient of the medium. [1/cm]
    :param rho: Lateral distance from the source. [cm]
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param d: Thickness of the slab. [cm]
    :param m_max: Maximum index of the truncated summation. The summation goes from -m_max to +m_max, inclusive, so
        the total number of terms is 2 * m_max + 1.
    :return: The normalized first-order autocorrelation function in reflectance. A vector the same length as tau.
    """
    return (g1_reflectance(msd, mua, musp, rho, n, lambda0, d, m_max) /
            g1_reflectance(0, mua, musp, rho, n, lambda0, d, m_max))


def d_factors_transmittance(msd0: np.ndarray, mua0: float, musp0: float, rho: float, n: float, lambda0: float,
                            d: float, m_max: int) -> tuple:
    """
    Calculates the d factors for the DCS Modified Beer-Lambert law for the laterally infinite slab in transmittance.
    See [1] for an explanation.

    [1] Baker, W. et al. (2014), "Modified Beer-Lambert law for blood flow"

    :param msd0: Baseline mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua0: Baseline absorption coefficient of the medium. [1/cm]
    :param musp0: Baseline reduced scattering coefficient of the medium. [1/cm]
    :param rho: Lateral distance from the source. [cm]
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param d: Thickness of the slab. [cm]
    :param m_max: Maximum index of the truncated summation. The summation goes from -m_max to +m_max, inclusive, so
        the total number of terms is 2 * m_max + 1.
    :return: The d factors. A tuple of three vectors (dr, da, ds), each the same length as tau. dr is the d factor for
        msd, da is the d factor for mua, and ds is the d factor for musp. To get the d factor for the dynamical
        parameter of interest (i.e., Db or v_ms), multiply dr by d(msd)/d(Db) or d(msd)/d(v_ms), respectively.
    """
    lambda0 *= 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp0
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp0)
    k = np.sqrt(3 * musp0 * mua0 + musp0**2 * k0**2 * msd0)
    dk = (3 * mua0 + musp0 * k0**2 * msd0) / (2 * k) # Derivative of k with respect to musp
    mu_eff = np.sqrt(3 * mua0 * musp0)
    dmu_eff = 3 * mua0 / (2 * mu_eff) # Derivative of mu_eff with respect to musp

    # dr, da
    num = 0
    num0 = 0
    den = 0
    den0 = 0
    for m in range(-m_max, m_max + 1):
        z1 = d * (1 - 2 * m) - 4 * m * zb - z0
        z2 = d * (1 - 2 * m) - (4 * m  - 2) * zb + z0
        r1 = np.sqrt(rho**2 + z1**2)
        r2 = np.sqrt(rho**2 + z2**2)
        term1_num = z1 / r1 * np.exp(-k * r1)
        term1_num_0 = z1 / r1 * np.exp(-mu_eff * r1)
        term2_num = z2 / r2 * np.exp(-k * r2)
        term2_num_0 = z2 / r2 * np.exp(-mu_eff * r2)
        term1_den = z1 / r1**3 * (1 + k * r1) * np.exp(-k * r1)
        term1_den_0 = z1 / r1**3 * (1 + mu_eff * r1) * np.exp(-mu_eff * r1)
        term2_den = z2 / r2**3 * (1 + k * r2) * np.exp(-k * r2)
        term2_den_0 = z2 / r2**3 * (1 + mu_eff * r2) * np.exp(-mu_eff * r2)

        num += term1_num - term2_num
        num0 += term1_num_0 - term2_num_0
        den += term1_den - term2_den
        den0 += term1_den_0 - term2_den_0

    dr = k0**2 * musp0**2 * num / den
    da = 3 * musp0 * (num / den - num0 / den0)

    # ds
    a1 = 0
    a1_0 = 0
    a2 = 0
    a2_0 = 0
    for m in range(-m_max, m_max + 1):
        z1 = d * (1 - 2 * m) - 4 * m * zb - z0
        z2 = d * (1 - 2 * m) - (4 * m  - 2) * zb + z0
        r1 = np.sqrt(rho**2 + z1**2)
        r2 = np.sqrt(rho**2 + z2**2)
        # Derivatives
        dz1 = z0 * (4 * m * zb + z0)
        dz2 = z0 * ((4 * m - 2) * zb - z0)
        dr1 = z1 / r1 * dz1
        dr2 = z2 / r2 * dz2

        a1 += z1 / r1**3 * np.exp(-k * r1) * ((dz1 / z1 - 3 * dr1 / r1) * (1 + k * r1) - k * r1 * (dk * r1 + k * dr1))
        a1_0 += z1 / r1**3 * np.exp(-mu_eff * r1) * ((dz1 / z1 - 3 * dr1 / r1) * (1 + mu_eff * r1)
                                                     - mu_eff * r1 * (dmu_eff * r1 + mu_eff * dr1))
        a2 += z2 / r2**3 * np.exp(-k * r2) * ((dz2 / z2 - 3 * dr2 / r2) * (1 + k * r2) - k * r2 * (dk * r2 + k * dr2))
        a2_0 += z2 / r2**3 * np.exp(-mu_eff * r2) * ((dz2 / z2 - 3 * dr2 / r2) * (1 + mu_eff * r2)
                                                     - mu_eff * r2 * (dmu_eff * r2 + mu_eff * dr2))

    ds = - 2 / den * (a1 - a2) + 2 / den0 * (a1_0 - a2_0)

    return dr, da, ds


def d_factors_reflectance(msd0: np.ndarray, mua0: float, musp0: float, rho: float, n: float, lambda0: float,
                          d: float, m_max: int) -> tuple:
    """
    Calculates the d factors for the DCS Modified Beer-Lambert law for the laterally infinite slab in reflectance.
    See [1] for an explanation.

    [1] Baker, W. et al. (2014), "Modified Beer-Lambert law for blood flow"

    :param msd0: Baseline mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua0: Baseline absorption coefficient of the medium. [1/cm]
    :param musp0: Baseline reduced scattering coefficient of the medium. [1/cm]
    :param rho: Lateral distance from the source. [cm]
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param d: Thickness of the slab. [cm]
    :param m_max: Maximum index of the truncated summation. The summation goes from -m_max to +m_max, inclusive, so
        the total number of terms is 2 * m_max + 1.
    :return: The d factors. A tuple of three vectors (dr, da, ds), each the same length as tau. dr is the d factor for
        msd, da is the d factor for mua, and ds is the d factor for musp. To get the d factor for the dynamical
        parameter of interest (i.e., Db or v_ms), multiply dr by d(msd)/d(Db) or d(msd)/d(v_ms), respectively.
    """
    lambda0 *= 1e-7  # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp0
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp0)
    k = np.sqrt(3 * musp0 * mua0 + musp0 ** 2 * k0 ** 2 * msd0)
    dk = (3 * mua0 + musp0 * k0 ** 2 * msd0) / (2 * k)  # Derivative of k with respect to musp
    mu_eff = np.sqrt(3 * mua0 * musp0)
    dmu_eff = 3 * mua0 / (2 * mu_eff)  # Derivative of mu_eff with respect to musp

    # dr, da
    num = 0
    num0 = 0
    den = 0
    den0 = 0
    for m in range(-m_max, m_max + 1):
        z3 = -2 * m * d - 4 * m * zb - z0
        z4 = -2 * m * d - (4 * m - 2) * zb + z0
        r3 = np.sqrt(rho ** 2 + z3 ** 2)
        r4 = np.sqrt(rho ** 2 + z4 ** 2)
        term1_num = z3 / r3 * np.exp(-k * r3)
        term1_num_0 = z3 / r3 * np.exp(-mu_eff * r3)
        term2_num = z4 / r4 * np.exp(-k * r4)
        term2_num_0 = z4 / r4 * np.exp(-mu_eff * r4)
        term1_den = z3 / r3 ** 3 * (1 + k * r3) * np.exp(-k * r3)
        term1_den_0 = z3 / r3 ** 3 * (1 + mu_eff * r3) * np.exp(-mu_eff * r3)
        term2_den = z4 / r4 ** 3 * (1 + k * r4) * np.exp(-k * r4)
        term2_den_0 = z4 / r4 ** 3 * (1 + mu_eff * r4) * np.exp(-mu_eff * r4)

        num += term1_num - term2_num
        num0 += term1_num_0 - term2_num_0
        den += term1_den - term2_den
        den0 += term1_den_0 - term2_den_0

    dr = k0 ** 2 * musp0 ** 2 * num / den
    da = 3 * musp0 * (num / den - num0 / den0)

    # ds
    a1 = 0
    a1_0 = 0
    a2 = 0
    a2_0 = 0
    for m in range(-m_max, m_max + 1):
        z3 = d * (1 - 2 * m) - 4 * m * zb - z0
        z4 = d * (1 - 2 * m) - (4 * m - 2) * zb + z0
        r3 = np.sqrt(rho ** 2 + z3 ** 2)
        r4 = np.sqrt(rho ** 2 + z4 ** 2)
        # Derivatives
        dz1 = z0 * (4 * m * zb + z0)
        dz2 = z0 * ((4 * m - 2) * zb - z0)
        dr1 = z3 / r3 * dz1
        dr2 = z4 / r4 * dz2

        a1 += z3 / r3 ** 3 * np.exp(-k * r3) * ((dz1 / z3 - 3 * dr1 / r3) * (1 + k * r3) - k * r3 * (dk * r3 + k * dr1))
        a1_0 += z3 / r3 ** 3 * np.exp(-mu_eff * r3) * ((dz1 / z3 - 3 * dr1 / r3) * (1 + mu_eff * r3)
                                                       - mu_eff * r3 * (dmu_eff * r3 + mu_eff * dr1))
        a2 += z4 / r4 ** 3 * np.exp(-k * r4) * ((dz2 / z4 - 3 * dr2 / r4) * (1 + k * r4) - k * r4 * (dk * r4 + k * dr2))
        a2_0 += z4 / r4 ** 3 * np.exp(-mu_eff * r4) * ((dz2 / z4 - 3 * dr2 / r4) * (1 + mu_eff * r4)
                                                       - mu_eff * r4 * (dmu_eff * r4 + mu_eff * dr2))

    ds = - 2 / den * (a1 - a2) + 2 / den0 * (a1_0 - a2_0)

    return dr, da, ds
