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


def g1_transmittance(msd: np.ndarray, mua: float, musp: float, l: tuple[float, float, float], rs: tuple[float, float],
                     rd: tuple[float, float], n: float, lambda0: float, m_max: tuple[int, int, int]) -> np.ndarray:
    """
    Calculates the unnormalized first-order autocorrelation function in transmittance for a parallelepiped.
    The z axis is the direction of the slab, and the x and y axes are the lateral dimensions, so the source is on the
    z=0 plane (or close to it if it is buried), and the detector is on the z=Lz plane.
    The prefactor, which depends on the source term, is set to 1 instead. Since only the normalized g1 is used, this is
    not a problem.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Scattering coefficient of the medium. [1/cm]
    :param l: Dimensions of the parallelepiped. A 3-tuple (lx, ly, lz) [cm].
    :param rs: Lateral position of the source. A 2-tuple (xs, ys). zs is set to 1 / musp [cm].
    :param rd: Lateral position of the detector. A 2-tuple (xd, yd). zd is set to Lz [cm].
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param m_max: Maximum indices (m_max_x, m_max_y, m_max_z) of the 3 truncated summations (one per axis).
        The 3 summations goes from -m_max to +m_max, inclusive, so the total number of terms is 2 * m_max + 1.
    :return: The unnormalized first-order autocorrelation function in transmittance. A vector the same length as tau.
    """
    # Unpack the input parameters
    lx, ly, lz = l
    xs, ys = rs
    zs = 1 / musp
    xd, yd = rd
    zd = lz
    m_max_x, m_max_y, m_max_z = m_max

    k0 = 2 * np.pi / (lambda0 * 1e-7) # Convert to cm
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp)
    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    def term(z: float, r: float) -> float:
        """
        Each of the 8 terms in the summation for the first-order autocorrelation function.

        :param z: Either z1 or z2 (see below)
        :param r: One of r1 to r8 (see below)
        :return: The value of the term
        """
        return (lz - z) * (k + 1 / r) * np.exp(-k * r) / (r**2)

    g1 = 0
    for mx in range(-m_max_x, m_max_x + 1):
        x1 = 2 * mx * lx + 4 * mx * zb + xs
        x2 = 2 * mx * lx + (4 * mx - 2) * zb - xs
        for my in range(-m_max_y, m_max_y + 1):
            y1 = 2 * my * ly + 4 * my * zb + ys
            y2 = 2 * my * ly + (4 * my - 2) * zb - ys
            for mz in range(-m_max_z, m_max_z + 1):
                z1 = 2 * mz * lz + 4 * mz * zb + zs
                z2 = 2 * mz * lz + (4 * mz - 2) * zb - zs

                r1 = np.sqrt((xd - x1)**2 + (yd - y1)**2 + (zd - z1)**2)
                r2 = np.sqrt((xd - x1)**2 + (yd - y1)**2 + (zd - z2)**2)
                r3 = np.sqrt((xd - x1)**2 + (yd - y2)**2 + (zd - z1)**2)
                r4 = np.sqrt((xd - x1)**2 + (yd - y2)**2 + (zd - z2)**2)
                r5 = np.sqrt((xd - x2)**2 + (yd - y1)**2 + (zd - z1)**2)
                r6 = np.sqrt((xd - x2)**2 + (yd - y1)**2 + (zd - z2)**2)
                r7 = np.sqrt((xd - x2)**2 + (yd - y2)**2 + (zd - z1)**2)
                r8 = np.sqrt((xd - x2)**2 + (yd - y2)**2 + (zd - z2)**2)

                g1 += (term(z1, r1) - term(z2, r2) - term(z1, r3) + term(z2, r4)
                       - term(z1, r5) + term(z2, r6) + term(z1, r7) - term(z2, r8))

    return g1


def g1_transmittance_norm(msd: np.ndarray, mua: float, musp: float, l: tuple[float, float, float],
                          rs: tuple[float, float], rd: tuple[float, float], n: float, lambda0: float,
                          m_max: tuple[int, int, int]) -> np.ndarray:
    """
    Calculates the normalized first-order autocorrelation function in transmittance for a parallelepiped.
    The z axis is the direction of the slab, and the x and y axes are the lateral dimensions, so the source is on the
    z=0 plane (or close to it if it is buried), and the detector is on the z=Lz plane.

    :param msd: Mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua: Absorption coefficient of the medium. [1/cm]
    :param musp: Scattering coefficient of the medium. [1/cm]
    :param l: Dimensions of the parallelepiped. A 3-tuple (lx, ly, lz) [cm].
    :param rs: Lateral position of the source. A 2-tuple (xs, ys). zs is set to 1 / musp [cm].
    :param rd: Lateral position of the detector. A 2-tuple (xd, yd). zd is set to Lz [cm].
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param m_max: Maximum indices (m_max_x, m_max_y, m_max_z) of the 3 truncated summations (one per axis).
        The 3 summations goes from -m_max to +m_max, inclusive, so the total number of terms is 2 * m_max + 1.
    :return: The normalized first-order autocorrelation function in transmittance. A vector the same length as tau.
    """
    return (g1_transmittance(msd, mua, musp, l, rs, rd, n, lambda0, m_max) /
            g1_transmittance(0, mua, musp, l, rs, rd, n, lambda0, m_max))


def d_factors_transmittance(msd0: np.ndarray, mua0: float, musp0: float, l: tuple[float, float, float],
                            rs: tuple[float, float], rd: tuple[float, float], n: float, lambda0: float,
                            m_max: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the d factors for the DCS Modified Beer-Lambert law for the parallelepiped in transmittance.
    See [1] for an explanation.

    [1] Baker, W. et al. (2014), "Modified Beer-Lambert law for blood flow"

    :param msd0: Baseline mean-square displacement of the scatterers. A vector the same length as tau. [cm^2]
    :param mua0: Baseline absorption coefficient of the medium. [1/cm]
    :param musp0: Baseline scattering coefficient of the medium. [1/cm]
    :param l: Dimensions of the parallelepiped. A 3-tuple (lx, ly, lz) [cm].
    :param rs: Lateral position of the source. A 2-tuple (xs, ys). zs is set to 1 / musp [cm].
    :param rd: Lateral position of the detector. A 2-tuple (xd, yd). zd is set to Lz [cm].
    :param n: Refractive index of the medium.
    :param lambda0: Wavelength of the light source. [nm]
    :param m_max: Maximum indices (m_max_x, m_max_y, m_max_z) of the 3 truncated summations (one per axis).
        The 3 summations goes from -m_max to +m_max, inclusive, so the total number of terms is 2 * m_max + 1.
    :return: The d factors. A tuple of three vectors (dr, da, ds), each the same length as tau. dr is the d factor for
        msd, da is the d factor for mua, and ds is the d factor for musp. To get the d factor for the dynamical
        parameter of interest (i.e., Db or v_ms), multiply dr by d(msd)/d(Db) or d(msd)/d(v_ms), respectively.
    """
    # Unpack the input parameters
    lx, ly, lz = l
    xs, ys = rs
    zs = 1 / musp0
    xd, yd = rd
    zd = lz
    m_max_x, m_max_y, m_max_z = m_max

    k0 = 2 * np.pi / (lambda0 * 1e-7) # Convert to cm
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp0)
    k = np.sqrt(3 * musp0 * mua0 + musp0**2 * k0**2 * msd0)
    mu_eff = np.sqrt(3 * musp0 * mua0)

    # Some derivatives
    dk_ds = (3 * mua0 + k0 ** 2 * musp0 * msd0) / (2 * k)
    dmu_eff_ds = 3 * mua0 / (2 * mu_eff)
    dk_da = 3 * musp0 / (2 * k)
    dmu_eff_da = 3 * musp0 / (2 * mu_eff)
    dk_dmsd = musp0**2 * k0**2 / (2 * k)
    dzs_ds = -zs ** 2
    dzb_ds = -zs * zb

    def dterm_dk(z: float, r: float, k: float | np.ndarray) -> float:
        """
        The derivative with respect to k of each of the 8 terms in the summation for the first-order
        autocorrelation function.

        :param z: Either z1 or z2 (see below)
        :param r: One of r1 to r8 (see below)
        :param k: The value of k for the current msd. This is either k or mu_eff, depending on the term.
        :return: The value of the term
        """
        return -(lz - z) * k * np.exp(-k * r) / (r**2)

    def dterm_ds(z: float, r: float, dz_ds: float, dr_ds: float, k: float | np.ndarray, dk_ds: float | np.ndarray) \
            -> float:
        """
        The derivative with respect to musp of each of the 8 terms in the summation for the first-order
        autocorrelation function.

        :param z: Either z1 or z2 (see below)
        :param r: One of r1 to r8 (see below)
        :param dz_ds: The derivative of z with respect to musp
        :param dr_ds: The derivative of r with respect to musp
        :param k: The value of k for the current msd. This is either k or mu_eff, depending on the term.
        :param dk_ds: The derivative of k with respect to musp. This is either dk_ds or dmu_eff_ds,
            depending on the term.
        :return: The value of the term
        """
        prefactor = np.exp(-k * r) / (r**2)
        term1 = -dz_ds * (k + 1 / r)
        term2 = (lz - z) * (dk_ds - 1 / r**2 * dr_ds)
        term3 = -(lz - z) * (k + 1 / r) * (dk_ds * r + (k + 2 / r) * dr_ds)
        return prefactor * (term1 + term2 + term3)

    # dr
    dg1_dk = 0
    dg10_dk = 0
    dg1_ds = 0
    dg10_ds = 0
    g1 = g1_transmittance(msd0, mua0, musp0, l, rs, rd, n, lambda0, m_max)
    g10 = g1_transmittance(0, mua0, musp0, l, rs, rd, n, lambda0, m_max)
    for mx in range(-m_max_x, m_max_x + 1):
        x1 = 2 * mx * lx + 4 * mx * zb + xs
        dx1_ds = 4 * mx * dzb_ds
        x2 = 2 * mx * lx + (4 * mx - 2) * zb - xs
        dx2_ds = (4 * mx - 2) * dzb_ds
        for my in range(-m_max_y, m_max_y + 1):
            y1 = 2 * my * ly + 4 * my * zb + ys
            dy1_ds = 4 * my * dzb_ds
            y2 = 2 * my * ly + (4 * my - 2) * zb - ys
            dy2_ds = (4 * my - 2) * dzb_ds
            for mz in range(-m_max_z, m_max_z + 1):
                z1 = 2 * mz * lz + 4 * mz * zb + zs
                dz1_ds = 4 * mz * dzb_ds + dzs_ds
                z2 = 2 * mz * lz + (4 * mz - 2) * zb - zs
                dz2_ds = (4 * mz - 2) * dzb_ds - dzs_ds

                r1 = np.sqrt((xd - x1)**2 + (yd - y1)**2 + (zd - z1)**2)
                r2 = np.sqrt((xd - x1)**2 + (yd - y1)**2 + (zd - z2)**2)
                r3 = np.sqrt((xd - x1)**2 + (yd - y2)**2 + (zd - z1)**2)
                r4 = np.sqrt((xd - x1)**2 + (yd - y2)**2 + (zd - z2)**2)
                r5 = np.sqrt((xd - x2)**2 + (yd - y1)**2 + (zd - z1)**2)
                r6 = np.sqrt((xd - x2)**2 + (yd - y1)**2 + (zd - z2)**2)
                r7 = np.sqrt((xd - x2)**2 + (yd - y2)**2 + (zd - z1)**2)
                r8 = np.sqrt((xd - x2)**2 + (yd - y2)**2 + (zd - z2)**2)
                dr1_ds = -1 / r1 * ((xd - x1) * dx1_ds + (yd - y1) * dy1_ds + (zd - z1) * dz1_ds)
                dr2_ds = -1 / r2 * ((xd - x1) * dx1_ds + (yd - y1) * dy1_ds + (zd - z2) * dz2_ds)
                dr3_ds = -1 / r3 * ((xd - x1) * dx1_ds + (yd - y2) * dy2_ds + (zd - z1) * dz1_ds)
                dr4_ds = -1 / r4 * ((xd - x1) * dx1_ds + (yd - y2) * dy2_ds + (zd - z2) * dz2_ds)
                dr5_ds = -1 / r5 * ((xd - x2) * dx2_ds + (yd - y1) * dy1_ds + (zd - z1) * dz1_ds)
                dr6_ds = -1 / r6 * ((xd - x2) * dx2_ds + (yd - y1) * dy1_ds + (zd - z2) * dz2_ds)
                dr7_ds = -1 / r7 * ((xd - x2) * dx2_ds + (yd - y2) * dy2_ds + (zd - z1) * dz1_ds)
                dr8_ds = -1 / r8 * ((xd - x2) * dx2_ds + (yd - y2) * dy2_ds + (zd - z2) * dz2_ds)

                dg1_dk += (dterm_dk(z1, r1, k) - dterm_dk(z2, r2, k) - dterm_dk(z1, r3, k) + dterm_dk(z2, r4, k) -
                           dterm_dk(z1, r5, k) + dterm_dk(z2, r6, k) + dterm_dk(z1, r7, k) - dterm_dk(z2, r8, k))
                dg10_dk += (dterm_dk(z1, r1, mu_eff) - dterm_dk(z2, r2, mu_eff) -
                            dterm_dk(z1, r3, mu_eff) + dterm_dk(z2, r4, mu_eff) -
                            dterm_dk(z1, r5, mu_eff) + dterm_dk(z2, r6, mu_eff) +
                            dterm_dk(z1, r7, mu_eff) - dterm_dk(z2, r8, mu_eff))

                dg1_ds += (dterm_ds(z1, r1, dz1_ds, dr1_ds, k, dk_ds) -
                           dterm_ds(z2, r2, dz2_ds, dr2_ds, k, dk_ds) -
                           dterm_ds(z1, r3, dz1_ds, dr3_ds, k, dk_ds) +
                           dterm_ds(z2, r4, dz2_ds, dr4_ds, k, dk_ds) -
                           dterm_ds(z1, r5, dz1_ds, dr5_ds, k, dk_ds) +
                           dterm_ds(z2, r6, dz2_ds, dr6_ds, k, dk_ds) +
                           dterm_ds(z1, r7, dz1_ds, dr7_ds, k, dk_ds) -
                           dterm_ds(z2, r8, dz2_ds, dr8_ds, k, dk_ds))
                dg10_ds += (dterm_ds(z1, r1, dz1_ds, dr1_ds, mu_eff, dmu_eff_ds) -
                            dterm_ds(z2, r2, dz2_ds, dr2_ds, mu_eff, dmu_eff_ds) -
                            dterm_ds(z1, r3, dz1_ds, dr3_ds, mu_eff, dmu_eff_ds) +
                            dterm_ds(z2, r4, dz2_ds, dr4_ds, mu_eff, dmu_eff_ds) -
                            dterm_ds(z1, r5, dz1_ds, dr5_ds, mu_eff, dmu_eff_ds) +
                            dterm_ds(z2, r6, dz2_ds, dr6_ds, mu_eff, dmu_eff_ds) +
                            dterm_ds(z1, r7, dz1_ds, dr7_ds, mu_eff, dmu_eff_ds) -
                            dterm_ds(z2, r8, dz2_ds, dr8_ds, mu_eff, dmu_eff_ds))

    dr = -2 * dk_dmsd * dg1_dk / g1
    da = -2 * (dk_da * dg1_dk / g1 - dmu_eff_da * dg10_dk / g10)
    ds = -2 * (dg1_ds / g1 - dg10_ds / g10)

    return dr, da, ds

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tau = np.logspace(-6, -2, 200)
    db = 1e-8
    msd = common.msd_brownian(tau, db)
    mua = 0.1
    musp = 10
    l = (1.0, 1.0, 1.0)
    rs = (0.5, 0.5)
    rd = (0.5, 0.5)
    n = 1.4
    lambda0 = 785
    m_max = (10, 10, 10)

    (dr, da, ds) = d_factors_transmittance(msd, mua, musp, l, rs, rd, n, lambda0, m_max)
    ddb = dr * common.d_msd_brownian(tau)

    # Plot ddb on right axis, da and ds on left axis
    fig, ax1 = plt.subplots()

    ax1.semilogx(tau, da, "--", label=r"$d_a$")
    ax1.semilogx(tau, ds, "--", label=r"$d_s$")
    ax1.set_xlabel(r"$\tau$ (s)")
    ax1.set_ylabel(r"$d_a, d_s$ (cm)")

    ax2 = ax1.twinx()
    ax2.semilogx(tau, ddb * 1e-8, label=r"$d_{Db}$", color='tab:green')
    ax2.set_ylabel(r"$d_{Db} \times 10^{-8}$ (s/cm$^2$)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Sensitivity factors for homogeneous parallelepiped (transmittance)")
    plt.show()
