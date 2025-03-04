import numpy as np
import forward.homogeneous_semi_inf as hsi


class MBLHomogeneousSemiInf:
    """
    Class for extracting Db from DCS data using the DCS Modified Beer-Lambert law for a homogeneous semi-infinite
    medium.
    """

    def __init__(
            self,
            tau: np.ndarray,
            g2_norm: np.ndarray,
            mua: np.ndarray | float,
            musp: np.ndarray | float,
            rho: float,
            n: float,
            lambda0: float,
            g2_norm_0: np.ndarray,
            mua0: float,
            musp0: float,
            db0: float,
    ):
        """
        Class constructor.

        :param tau: Vector of time delays [s].
        :param g2_norm: Matrix of normalized second-order autocorrelation functions. Each column corresponds to a
            different measurement, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param mua: Absorption coefficient of the medium [1/cm]. If a float, the same value is used for all
            measurements. If an array, a different value is used for each measurement, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param musp: Reduced scattering coefficient of the medium [1/cm]. If a float, the same value is used for all
            measurements. If an array, a different value is used for each measurement, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param rho: Source-detector separation [cm].
        :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
            (typically air).
        :param lambda0: Wavelength of the light source [nm].
        :param g2_norm_0: Normalized second-order autocorrelation function for the baseline measurement.
        :param mua0: Absorption coefficient of the medium for the baseline measurement [1/cm].
        :param musp0: Reduced scattering coefficient of the medium for the baseline measurement [1/cm].
        :param db0: Brownian motion diffusion coefficient for the baseline measurement [cm^2/s].
        """
        # Check that the number of rows in g2_norm is the same as the length of tau
        if g2_norm.shape[0] == len(tau):
            self.tau = tau
            self.g2_norm = g2_norm
        else:
            raise ValueError("The number of rows in g2_norm should be the same as the length of tau")

        # Check that mua and musp are either floats or arrays of the same length as the number of columns in g2_norm
        if isinstance(mua, (float, int)):
            self.mua = np.full(len(self), mua)
        elif isinstance(mua, np.ndarray):
            if len(mua) == len(self):
                self.mua = mua
            else:
                raise ValueError("mua should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")
        else:
            raise ValueError("mua should be a float or an array")
        if isinstance(musp, (float, int)):
            self.musp = np.full(len(self), musp)
        elif isinstance(musp, np.ndarray):
            if len(musp) == len(self):
                self.musp = musp
            else:
                raise ValueError("musp should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")

        # Check that the length of g2_norm_0 is the same as the length of tau
        if len(g2_norm_0) == len(tau):
            self.g2_norm_0 = g2_norm_0
        else:
            raise ValueError("g2_norm_0 should be the same length as tau")

        self.rho = rho
        self.n = n
        self.lambda0 = lambda0
        self.mua0 = mua0
        self.musp0 = musp0
        self.db0 = db0

    def __len__(self):
        """
        Returns the number of measurements.
        """
        return self.g2_norm.shape[1]

    def fit(self) -> np.ndarray:
        """
        Uses the DCS Modified Beer-Lambert law to calculate the Brownian motion diffusion coefficient for each lag time
        and measurement.
        :return: The calculated Brownian motion diffusion coefficients. A matrix the same size as g2_norm.
        """
        # Calculate variations in mua and musp from the baseline
        delta_mua = self.mua - self.mua0
        delta_musp = self.musp - self.musp0

        # Calculate the d factors for the baseline and broadcast them to the same shape as g2_norm
        (df, da, ds) = hsi.d_factors(self.db0, self.tau, self.mua0, self.musp0, self.rho, self.n, self.lambda0)
        df = np.expand_dims(df, axis=1)
        df = np.broadcast_to(df, self.g2_norm.shape)
        da = np.expand_dims(da, axis=1)
        da = np.broadcast_to(da, self.g2_norm.shape)
        ds = np.expand_dims(ds, axis=1)
        ds = np.broadcast_to(ds, self.g2_norm.shape)

        # Calculate db
        g2_norm_0 = np.expand_dims(self.g2_norm_0, axis=1)
        g2_norm_0 = np.broadcast_to(g2_norm_0, self.g2_norm.shape)
        delta_od = -np.log((self.g2_norm - 1) / (g2_norm_0 - 1))
        delta_db = (delta_od - da * delta_mua - ds * delta_musp) / df
        db = self.db0 + delta_db

        return db
