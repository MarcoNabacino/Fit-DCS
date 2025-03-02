from typing import Dict

import numpy as np
import forward.common as common
import forward.homogeneous_semi_inf as hsi


class MSDModel:
    """
    A class for calculating the mean-square displacement (MSD) of the scatterers based on a model.
    """

    def __init__(self, model_name: str, param_init: Dict, param_bounds: Dict | None = None):
        """
        Class constructor.

        :param model_name: The name of the MSD model.
        :param param_init: Initial values of the parameters of the model. A dictionary with the parameter names as keys
            and the initial values as values.
        :param param_bounds: Bounds of the parameters of the model. A dictionary with the parameter names as keys and
            2-tuples as values, where the first element is the lower bound and the second element is the upper bound.
        """
        self.model_name = model_name
        self._get_msd_fn()

        # Check that param_init contains initial values for all parameters of the model, that is, the keys of param_init
        # should match the params of the model.
        if set(param_init.keys()) == set(self.params):
            self.param_init = param_init
        else:
            raise ValueError("param_init should contain initial values for all parameters of the model")

        # Check that, if param_bounds is not None, then it contains bounds for all parameters of the model, that is,
        # the keys of param_bounds should match the params of the model.
        if param_bounds is not None:
            if set(param_bounds.keys()) == set(self.params):
                self.param_bounds = param_bounds
            else:
                raise ValueError("param_bounds should contain bounds for all parameters of the model")

    def _get_msd_fn(self):
        """
        Sets the mean-square displacement function based on the model name, as well as the parameters of the model that
        will be fit.
        """
        if self.model_name == "brownian":
            msd_fn = common.msd_brownian
            params = ["db"]
        elif self.model_name == "ballistic":
            msd_fn = common.msd_ballistic
            params = ["v_ms"]
        elif self.model_name == "hybrid":
            msd_fn = common.msd_hybrid
            params = ["db", "v_ms"]
        else:
            raise ValueError(f"Unknown MSD model: {self.model_name}")

        self.msd_fn = msd_fn
        self.params = params


class BetaCalculation:
    """
    Class defining how to calculate the beta parameter.
    """

    def __init__(
            self,
            mode: str,
            tau_lims: tuple[float, float] | None = None,
            init: float | None = None,
            bounds: tuple[float, float] | None = None,
    ):
        """
        Class constructor.

        :param mode: "raw" or "fit". If "raw", the beta parameter is calculated from the raw data as
            beta = g2_norm(0) - 1, where g2_norm(0) is estimated as mean(g2_norm[tau_lims[0]:tau_lims[1]]).
            If "fit", the beta parameter is fit to g2_norm using the initial value init and the bounds bounds.
        :param tau_lims: Lower and upper limits of the tau values to use for calculating g2_norm(0) [s].
            Only used if mode is "raw".
        :param init: Initial value of the beta parameter. Only used if mode is "fit".
        :param bounds: Bounds of the beta parameter. Only used if mode is "fit".
        """
        # Check that the mode is either "raw" or "fit"
        if mode in ["raw", "fit"]:
            self.mode = mode
        else:
            raise ValueError("mode should be either 'raw' or 'fit'")

        # Check that, if mode is "raw", then tau_lims is an ordered pair of floats
        if mode == "raw":
            if tau_lims is not None:
                if len(tau_lims) == 2 and tau_lims[0] < tau_lims[1]:
                    self.tau_lims = tau_lims
                else:
                    raise ValueError("tau_lims should be an ordered pair of floats")
            else:
                raise ValueError("tau_lims should be provided if mode is 'raw'")
        # Check that, if mode is "fit", then init is a float and bounds is an ordered pair of floats
        elif mode == "fit":
            if init is not None:
                if isinstance(init, float):
                    self.init = init
                else:
                    raise ValueError("init should be a float")
            else:
                raise ValueError("init should be provided if mode is 'fit'")

            if bounds is not None:
                if len(bounds) == 2 and bounds[0] < bounds[1]:
                    self.bounds = bounds
                else:
                    raise ValueError("bounds should be an ordered pair of floats")
            else:
                raise ValueError("bounds should be provided if mode is 'fit'")


class FitHomogeneousSemiInf:
    """
    A class for fitting the normalized second-order autocorrelation functions g2_norm using the homogeneous
    semi-infinite model.
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
            msd_model: MSDModel,
            beta_calculation: BetaCalculation,
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
        :param msd_model: An instance of the MSDModel class.
        :param beta_calculation: An instance of the BetaCalculation class.
        """

        # Check that the number of rows in g2_norm is the same as the length of tau
        if g2_norm.shape[0] == len(tau):
            self.tau = tau
            self.g2_norm = g2_norm
        else:
            raise ValueError("The number of rows in g2_norm should be the same as the length of tau")

        # Check that mua and musp are either floats or arrays of the same length as the number of columns in g2_norm
        if isinstance(mua, (float, int)):
            self.mua = np.full(g2_norm.shape[1], mua)
        elif isinstance(mua, np.ndarray):
            if len(mua) == g2_norm.shape[1]:
                self.mua = mua
            else:
                raise ValueError("mua should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")
        else:
            raise ValueError("mua should be a float or an array")
        if isinstance(musp, (float, int)):
            self.musp = np.full(g2_norm.shape[1], musp)
        elif isinstance(musp, np.ndarray):
            if len(musp) == g2_norm.shape[1]:
                self.musp = musp
            else:
                raise ValueError("musp should be a float or an array of the same length as the number of columns in "
                                 "g2_norm")

        self.rho = rho
        self.n = n
        self.lambda0 = lambda0
        self.msd_model = msd_model
        self.beta_calculation = beta_calculation


    def __len__(self):
        """
        Returns the number of measurements.
        """
        return self.g2_norm.shape[1]