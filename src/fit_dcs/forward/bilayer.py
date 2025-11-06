"""
 Fit-DCS: A Python toolbox for Diffuse Correlation Spectroscopy analysis
 Copyright (C) 2025  Marco Nabacino

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import fit_dcs.forward.common as common
from scipy import LowLevelCallable
from scipy.integrate import quad, quad_vec
from scipy.special import jv
from fit_dcs.core.lib_loader import BILAYER_LIB
import ctypes
from ctypes import c_double, c_void_p


def g1_spatial_freq(msd: np.ndarray, mua_up: float, mua_dn: float,
                    musp_up: float, musp_dn: float,
                    n: float, d: float, lambda0: float, q: float) -> np.ndarray:
    """
    Calculates the unnormalized G1 in the spatial frequency domain for a bilayer medium. From [1].

    [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: Theoretical framework,
    recent advances in hardware, analysis, and applications."

    :param msd: Mean-square displacement in both layers. Shape (2, len(tau)) where first row
        is upper layer and second row is lower layer [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
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

    xi_up = np.sqrt(q**2 + 3 * mua_up * musp_up + k0**2 * musp_up**2 * msd[0, :])
    xi_dn = np.sqrt(q**2 + 3 * mua_dn * musp_dn + k0**2 * musp_dn**2 * msd[1, :])

    term1 = 3 * musp_up * np.sinh(xi_up * (z0 + zb)) / xi_up
    num = xi_up * np.cosh(xi_up * d) / (3 * musp_up) + xi_dn * np.sinh(xi_up * d) / (3 * musp_dn)
    den = xi_up * np.cosh(xi_up * (d + zb)) / (3 * musp_up) + xi_dn * np.sinh(xi_up * (d + zb)) / (3 * musp_dn)
    term2 = 3 * musp_up * np.sinh(xi_up * z0) / xi_up

    return term1 * (num / den) - term2


def g1(msd: np.ndarray, mua_up: float, mua_dn: float,
       musp_up: float, musp_dn: float,
       n: float, d: float, rho: float, lambda0: float, q_max: float) -> np.ndarray:
    """
    Calculates the unnormalized G1 for a bilayer model. From [1].

    [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: Theoretical framework,
    recent advances in hardware, analysis, and applications."

    :param msd: Mean-square displacement in both layers. Shape (2, len(tau)) where first row
        is upper layer and second row is lower layer [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
    :param musp_dn: Reduced scattering coefficient of the lower layer. [1/cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param d: Thickness of the upper layer. [cm]
    :param rho: Source-detector separation. [cm]
    :param lambda0: Wavelength of the light source. [nm]
    :param q_max: Maximum spatial frequency for the integration. [1/cm]
    :return: The unnormalized first-order autocorrelation function G1. A vector the same length as tau.
    """
    n_tau_bins = msd.shape[1]
    if BILAYER_LIB is not None:
        result = np.empty(n_tau_bins)
        params = (c_double * 10)(
            0.0,  # Placeholder for msd_up
            0.0,  # Placeholder for msd_dn
            mua_up,
            mua_dn,
            musp_up,
            musp_dn,
            n,
            d,
            lambda0,
            rho
        )
        params_ptr = ctypes.cast(params, c_void_p)
        integrand = LowLevelCallable(BILAYER_LIB.integrand, params_ptr)

        for i in range(n_tau_bins):
            # Update msd values in params and integrate
            params[0] = float(msd[0, i])
            params[1] = float(msd[1, i])
            result[i], _ = quad(integrand, 0, q_max)
    else:
        def integrand(q):
            return (g1_spatial_freq(msd, mua_up, mua_dn, musp_up, musp_dn, n, d, lambda0, q)
                    * q * jv(0, q * rho))
        result, _ = quad_vec(integrand, 0, q_max)

    return result


def g1_norm(msd: np.ndarray, mua_up: float, mua_dn: float,
            musp_up: float, musp_dn: float,
            n: float, d: float, rho: float, lambda0: float, q_max: float) -> np.ndarray:
    """
    Calculates the normalized G1 for a bilayer model.

    :param msd: Mean-square displacement in both layers. Shape (2, len(tau)) where first row
        is upper layer and second row is lower layer [cm^2]
    :param mua_up: Absorption coefficient of the upper layer. [1/cm]
    :param mua_dn: Absorption coefficient of the lower layer. [1/cm]
    :param musp_up: Reduced scattering coefficient of the upper layer. [1/cm]
    :param musp_dn: Reduced scattering coefficient of the lower layer. [1/cm]
    :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
        (typically air).
    :param d: Thickness of the upper layer. [cm]
    :param rho: Source-detector separation. [cm]
    :param lambda0: Wavelength of the light source. [nm]
    :param q_max: Maximum spatial frequency for the integration. [1/cm]
    :return: The normalized first-order autocorrelation function g1. A vector the same length as tau.
    """
    norm = g1(np.zeros_like(msd), mua_up, mua_dn, musp_up, musp_dn, n, d, rho, lambda0, q_max)
    return g1(msd, mua_up, mua_dn, musp_up, musp_dn, n, d, rho, lambda0, q_max) / norm


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
    msd = np.vstack([msd_up, msd_dn])

    q_max_list = [20, 40, 60, 80, 100]

    for q_max in q_max_list:
        g2_norm = 1 + beta * g1_norm(msd, mua_up, mua_dn, musp_up, musp_dn, n, d, rho, lambda0, q_max) ** 2
        plt.semilogx(tau, g2_norm, label=q_max)
    plt.xlabel(r"$\tau$ (s)")
    plt.ylabel(r"$g_2$")
    plt.legend(title=r"$q_{max}$ (1/cm)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
