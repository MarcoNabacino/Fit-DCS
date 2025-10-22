import numpy as np
import fit_dcs.forward.common as common
from scipy import LowLevelCallable
from scipy.integrate import quad, quad_vec
from scipy.special import jv
from fit_dcs.core.lib_loader import BILAYER_LIB, _HAS_BILAYER_LIB
import ctypes
from ctypes import c_double, c_void_p


def g1_spatial_freq(msd_up: np.ndarray, mua_up: float, musp_up: float,
                    msd_dn: np.ndarray, mua_dn: float, musp_dn: float,
                    n: float, d: float, lambda0: float, q: float) -> np.ndarray:
    """
    Calculates the unnormalized G1 in the spatial frequency domain for a bilayer medium. From [1].

    [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: Theoretical framework,
    recent advances in hardware, analysis, and applications."

    :param msd_up: Mean-square displacement in the upper layer. A vector the same length as tau [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
    :param msd_dn: Mean-square displacement in the lower layer. A vector the same length as tau [cm^2]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_dn: Reduced scattering coefficient of the lower layer. [1/cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param d: Thickness of the upper layer. [cm]
    :param lambda0: Wavelength of the light source. [nm]
    :param q: Spatial frequency. [1/cm]
    :return: The unnormalized first-order autocorrelation function G1 in the spatial frequency domain.
        A vector the same length as tau.
    """
    lambda0 *= 1e-7  # Convert to cm
    k0 = 2 * np.pi / lambda0
    z0 = 1 / musp_up
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp_up)

    xi_up = np.sqrt(q**2 + 3 * mua_up * musp_up + k0**2 * musp_up**2 * msd_up)
    xi_dn = np.sqrt(q**2 + 3 * mua_dn * musp_dn + k0**2 * musp_dn**2 * msd_dn)

    term1 = 3 * musp_up * np.sinh(xi_up * (z0 + zb)) / xi_up
    num = xi_up * np.cosh(xi_up * d) / (3 * musp_up) + xi_dn * np.sinh(xi_up * d) / (3 * musp_dn)
    den = xi_up * np.cosh(xi_up * (d + zb)) / (3 * musp_up) + xi_dn * np.sinh(xi_up * (d + zb)) / (3 * musp_dn)
    term2 = 3 * musp_up * np.sinh(xi_up * z0) / xi_up

    return term1 * (num / den) - term2


def g1(msd_up: np.ndarray, mua_up: float, musp_up: float,
       msd_dn: np.ndarray, mua_dn: float, musp_dn: float,
       n: float, d: float, rho: float, lambda0: float, q_max: float) -> np.ndarray:
    """
    Calculates the unnormalized G1 for a bilayer model. From [1].

    [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: Theoretical framework,
    recent advances in hardware, analysis, and applications."

    :param msd_up: Mean-square displacement in the upper layer. A vector the same length as tau [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
    :param msd_dn: Mean-square displacement in the lower layer. A vector the same length as tau [cm^2]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_dn: Reduced scattering coefficient of the lower layer. [1/cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param d: Thickness of the upper layer. [cm]
    :param rho: Source-detector separation. [cm]
    :param lambda0: Wavelength of the light source. [nm]
    :param q_max: Maximum spatial frequency for the integration. [1/cm]
    :return: The unnormalized first-order autocorrelation function G1. A vector the same length as tau.
    """
    if _HAS_BILAYER_LIB:
        result = np.zeros_like(msd_up)
        params = (c_double * 10)(
            0.0,  # Placeholder for msd_up
            mua_up,
            musp_up,
            0.0,  # Placeholder for msd_dn
            mua_dn,
            musp_dn,
            n,
            d,
            lambda0,
            rho
        )
        params_ptr = ctypes.cast(params, c_void_p)
        integrand = LowLevelCallable(BILAYER_LIB.integrand, params_ptr)

        for i in range(len(msd_up)):
            # Update msd values in params and integrate
            params[0] = float(msd_up[i])
            params[3] = float(msd_dn[i])
            result[i], _ = quad(integrand, 0, q_max)
    else:
        def integrand(q):
            return (g1_spatial_freq(msd_up, mua_up, musp_up, msd_dn, mua_dn, musp_dn, n, d, lambda0, q)
                    * q * jv(0, q * rho))
        result, _ = quad_vec(integrand, 0, q_max)

    return result


def g1_norm(msd_up: np.ndarray, mua_up: float, musp_up: float,
            msd_dn: np.ndarray, mua_dn: float, musp_dn: float,
            n: float, d: float, rho: float, lambda0: float, q_max: float) -> np.ndarray:
    """
    Calculates the normalized G1 for a bilayer model.

    :param msd_up: Mean-square displacement in the upper layer. A vector the same length as tau [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
    :param msd_dn: Mean-square displacement in the lower layer. A vector the same length as tau [cm^2]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_dn: Reduced scattering coefficient of the lower layer. [1/cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param d: Thickness of the upper layer. [cm]
    :param rho: Source-detector separation. [cm]
    :param lambda0: Wavelength of the light source. [nm]
    :return: The normalized first-order autocorrelation function G1. A vector the same length as tau.
    """
    zero_vec = np.zeros_like(msd_up)
    norm = g1(zero_vec, mua_up, musp_up, zero_vec, mua_dn, musp_dn, n, d, rho, lambda0, q_max)
    return g1(msd_up, mua_up, musp_up, msd_dn, mua_dn, musp_dn, n, d, rho, lambda0, q_max) / norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fit_dcs.forward.common import msd_brownian

    db_up = 1e-8
    mua_up = 0.04
    musp_up = 10
    d = 1.0
    db_dn = 6e-8
    mua_dn = 0.2
    musp_dn = 10
    n = 1.4
    lambda0 = 785
    rho = 2.5
    beta = 0.5
    tau = np.logspace(-7, -2, 200)
    msd_up = msd_brownian(tau, db_up)
    msd_dn = msd_brownian(tau, db_dn)

    q_max_list = [20, 40, 60, 80, 100]

    for q_max in q_max_list:
        g2_norm = 1 + beta * g1_norm(msd_up, mua_up, musp_up, msd_dn, mua_dn, musp_dn, n, d, rho, lambda0, q_max) ** 2
        plt.semilogx(tau, g2_norm, label=q_max)
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel(r"$g_2$")
    plt.legend(title=r"$q_{max}$ (1/cm)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
