import numpy as np
import forward.common as common


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

    lambda0 *= 1e-7 # Convert to cm
    k0 = 2 * np.pi / lambda0
    a = common.a_coefficient_boundary(n)
    zb = 2 * a / (3 * musp)
    k = np.sqrt(3 * musp * mua + musp**2 * k0**2 * msd)

    def term(z: float, r: float):
        """
        Each of the 8 terms in the summation for the first-order autocorrelation function.
        :param z: Either z1 or z2 (see below)
        :param r: One of r1 to r8 (see below)
        :return: The value of the term.
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
