import numpy as np
import forward.common as common
from typing import Callable, Dict


class MSDModelMBL:
    """
    A class for specifying the mean-square displacement (MSD) model for analyzing the normalized second-order
    autocorrelation functions g2_norm using the Modified Beer-Lambert law. The possible models are:

    - "brownian": Brownian motion model with a single parameter, the diffusion coefficient (db).

    - "ballistic": ballistic motion model with a single parameter, the mean square speed of the
        scatterers (v_ms).
    """

    def __init__(self, name: str, param0: float):
        """
        Class constructor.

        :param name: The name of the model to use for the mean-square displacement.
            Choose between "brownian" or "ballistic".
        :param param0: The baseline value of the parameter, typically estimated previously via a fit. If model is
            "brownian", this is the baseline diffusion coefficient db [cm^2/s]. If model is "ballistic", this is the
            baseline mean square speed v_ms [cm/s].
        """
        # Check the model and fetch the appropriate function and derivative. Also check that the baseline parameters
        # are provided.
        if name == "brownian":
            self.msd_fn = common.msd_brownian
            self.d_msd_fn = common.d_msd_brownian
        elif name == "ballistic":
            self.msd_fn = common.msd_ballistic
            self.d_msd_fn = common.d_msd_ballistic
        else:
            raise ValueError(f"Model {name} not recognized. Choose between 'brownian' or 'ballistic'.")
        self.model = name
        self.param0 = param0


class MBLHomogeneous:
    """
    Class for extracting Db from DCS data using the DCS Modified Beer-Lambert law for a homogeneous semi-infinite
    medium.
    """

    def __init__(
            self,
            tau: np.ndarray,
            g2_norm: np.ndarray,
            g2_norm_0: np.ndarray,
            d_factors_fn: Callable,
            msd_model: MSDModelMBL,
            mua: np.ndarray | float,
            musp: np.ndarray | float,
            **kwargs
    ):
        """
        Class constructor.

        :param tau: Vector of time delays [s].
        :param g2_norm: Matrix of normalized second-order autocorrelation functions. Each column corresponds to a
            different measurement, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param g2_norm_0: Normalized second-order autocorrelation function for the baseline measurement. A vector the
            same length as tau.
        :param d_factors_fn: The function to use to calculate the d factors for the DCS Modified Beer-Lambert law. This
            typically comes from the forward module.
        :param msd_model: An instance of the MSDModelMBL class that specifies the mean-square displacement model to use.
        :param mua: Absorption coefficient of the medium [1/cm]. If a float, the same value is used for all
            measurements. If an array, a different value is used for each measurement, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param musp: Reduced scattering coefficient of the medium [1/cm]. If a float, the same value is used for all
            measurements. If an array, a different value is used for each measurement, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param kwargs: Arguments to be passed to d_factors_fn, which gets called as d_factors_fn(msd0, **kwargs), where
            msd0 gets calculated using the baseline parameter provided in msd_model.
        """
        # Check that the number of rows in g2_norm is the same as the length of tau
        if g2_norm.shape[0] == len(tau):
            self.tau = tau
            self.g2_norm = g2_norm
        else:
            raise ValueError("The number of rows in g2_norm should be the same as the length of tau")

        # Check that the length of g2_norm_0 is the same as the length of tau
        if len(g2_norm_0) == len(tau):
            self.g2_norm_0 = g2_norm_0
        else:
            raise ValueError("g2_norm_0 should be the same length as tau")

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
        else:
            raise ValueError("musp should be a float or an array")

        self.d_factors_fn = d_factors_fn
        self.msd_model = msd_model
        self.d_factors_fn_args = kwargs

    def __len__(self):
        """
        Returns the number of measurements.
        """
        return self.g2_norm.shape[1]

    def fit(self) -> np.ndarray:
        """
        Uses the DCS Modified Beer-Lambert law to calculate the msd parameter (Brownian diffusion coefficient or mean
        square velocity, depending on the model) for each lag time and iteration.

        :return: The calculated parameters for each lag time and iteration. A matrix the same size as g2_norm.
        """
        # Calculate variations in mua and musp from the baseline
        delta_mua = self.mua - self.d_factors_fn_args["mua0"]
        delta_musp = self.musp - self.d_factors_fn_args["musp0"]

        # Calculate the d factors for the baseline and broadcast them to the same shape as g2_norm
        msd0 = self.msd_model.msd_fn(self.tau, self.msd_model.param0)
        (dr, da, ds) = self.d_factors_fn(msd0, **self.d_factors_fn_args)
        # Calculate the d factor with respect to the parameter of interest based on dr and the derivative of the MSD
        dp = dr * self.msd_model.d_msd_fn(self.tau)
        # Broadcast the d factors to the same shape as g2_norm
        dp = np.expand_dims(dp, axis=1)
        dp = np.broadcast_to(dp, self.g2_norm.shape)
        da = np.expand_dims(da, axis=1)
        da = np.broadcast_to(da, self.g2_norm.shape)
        ds = np.expand_dims(ds, axis=1)
        ds = np.broadcast_to(ds, self.g2_norm.shape)

        # Calculate parameter of interest (Db or v_ms) for each measurement and lag time.
        g2_norm_0 = np.expand_dims(self.g2_norm_0, axis=1)
        g2_norm_0 = np.broadcast_to(g2_norm_0, self.g2_norm.shape)
        delta_od = -np.log((self.g2_norm - 1) / (g2_norm_0 - 1))
        delta_param = (delta_od - da * delta_mua - ds * delta_musp) / dp
        param = self.msd_model.param0 + delta_param

        return param
