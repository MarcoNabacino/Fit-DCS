from typing import Dict

import numpy as np
import forward.common as common
import forward.homogeneous_semi_inf as hsi
import scipy.optimize as opt


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
        :param param_bounds: Bounds of the parameters of the model (optional). A dictionary with the parameter names as
            keys and 2-tuples as values, where the first element is the lower bound and the second element is the upper
            bound. To set no bound, use None rather that inf. If None, no bounds are set.
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
        else:
            self.param_bounds = {param: (None, None) for param in self.params}

    def _get_msd_fn(self):
        """
        Sets the mean-square displacement function based on the model name, as well as the parameters of the model that
        will be fit and their scale.
        """
        if self.model_name == "brownian":
            msd_fn = common.msd_brownian
            params = ["db"]
            params_scale = {"db": 1e-8}
        elif self.model_name == "ballistic":
            msd_fn = common.msd_ballistic
            params = ["v_ms"]
            params_scale = {"v_ms": 1e-4}
        elif self.model_name == "hybrid":
            msd_fn = common.msd_hybrid
            params = ["db", "v_ms"]
            params_scale = {"db": 1e-8, "v_ms": 1e-4}
        else:
            raise ValueError(f"Unknown MSD model: {self.model_name}")

        self.msd_fn = msd_fn
        self.params = params
        self.params_scale = params_scale


class BetaCalculator:
    """
    Class defining how to calculate the beta parameter.
    """

    def __init__(self, mode: str, **kwargs):
        """
        Class constructor.

        :param mode: The mode of the beta calculation. Either "fixed", "raw", or "fit".
        :param kwargs: Additional arguments depending on the mode. Specific arguments are:
            - If mode is "fixed", then beta_fixed should be a float.
            - If mode is "raw", then tau_lims should be an ordered pair of floats.
            - If mode is "fit", then beta_init should be a float and beta_bounds (optional) should be an ordered
                pair of floats or None.
        """

        # Check mode and set the corresponding attributes
        if mode == "fixed":
            if "beta_fixed" in kwargs:
                # Check that beta_fixed is a float
                if isinstance(kwargs["beta_fixed"], (float, int)):
                    self.beta_fixed = kwargs["beta_fixed"]
                    # Warn the user if beta_fixed is outside the range [0, 0.5]
                    if self.beta_fixed < 0 or self.beta_fixed > 0.5:
                        print("Warning: beta_fixed should be in the range [0, 0.5]")
                else:
                    raise ValueError("beta_fixed should be a float")
            else:
                raise ValueError("beta_fixed should be provided for the 'fixed' mode")
        elif mode == "raw":
            if "tau_lims" in kwargs:
                # Check that tau_lims is an ordered pair of floats
                if isinstance(kwargs["tau_lims"], tuple) and len(kwargs["tau_lims"]) == 2:
                    if (all(isinstance(x, (float, int)) for x in kwargs["tau_lims"])) and kwargs["tau_lims"][0] < \
                            kwargs["tau_lims"][1]:
                        self.tau_lims = kwargs["tau_lims"]
                    else:
                        raise ValueError("tau_lims should be an ordered pair of floats")
                else:
                    raise ValueError("tau_lims should be an ordered pair of floats")
            else:
                raise ValueError("tau_lims should be provided for the 'raw' mode")
        elif mode == "fit":
            if "beta_init" in kwargs:
                # Check that beta_init is a float
                if isinstance(kwargs["beta_init"], (float, int)):
                    self.beta_init = kwargs["beta_init"]
                else:
                    raise ValueError("beta_init should be a float")
            else:
                raise ValueError("beta_init should be provided for the 'fit' mode")
            if "beta_bounds" in kwargs:
                # Check that beta_bounds is an ordered pair of floats
                if isinstance(kwargs["beta_bounds"], tuple) and len(kwargs["beta_bounds"]) == 2:
                    if (all(isinstance(x, (float, int)) for x in kwargs["beta_bounds"])) and kwargs["beta_bounds"][0] < \
                            kwargs["beta_bounds"][1]:
                        self.beta_bounds = kwargs["beta_bounds"]
                    else:
                        raise ValueError("beta_bounds should be an ordered pair of floats")
                else:
                    raise ValueError("beta_bounds should be an ordered pair of floats")
            else:
                self.beta_bounds = None
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'fixed', 'raw', or 'fit'")

        self.mode = mode


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
            beta_calculator: BetaCalculator,
            tau_lims_fit: tuple | None = None,
            g1_lim_fit: float | None = None
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
        :param beta_calculator: An instance of the BetaCalculator class.
        :param tau_lims_fit: Ordered pair of floats defining the lower and upper limits of the time delays used for
            fitting. If None, the entire tau range is used.
        :param g1_lim_fit: If not None, the portion of the g1_norm curve that is used for fitting is limited to the
            values greater than g1_lim_fit. If both tau_lims_fit and g1_lim_fit are provided, the fitting is done
            starting from tau_lims_fit[0] and up to the minimum of tau_lims_fit[1] and the first time delay where
            g1_norm is greater than g1_lim_fit.
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

        # Check that tau_lims_fit is an ordered pair of floats
        if tau_lims_fit is not None:
            if isinstance(tau_lims_fit, tuple) and len(tau_lims_fit) == 2:
                if (all(isinstance(x, (float, int)) for x in tau_lims_fit)) and tau_lims_fit[0] < tau_lims_fit[1]:
                    self.tau_lims_fit = tau_lims_fit
                else:
                    raise ValueError("tau_lims_fit should be an ordered pair of floats")
            else:
                raise ValueError("tau_lims_fit should be an ordered pair of floats")
        else:
            self.tau_lims_fit = None

        self.rho = rho
        self.n = n
        self.lambda0 = lambda0
        self.msd_model = msd_model
        self.beta_calculator = beta_calculator
        self.g1_lim_fit = g1_lim_fit

        # Initialize the beta and fitted_params attribute to None
        self.beta = None
        self.fitted_params = None

    def __len__(self):
        """
        Returns the number of measurements.
        """
        return self.g2_norm.shape[1]

    def fit(self):
        """
        Fits the model to the data
        """

        if self.beta_calculator.mode in ["fixed", "raw"]:
            # Initialize arrays to store the fitted parameters values
            self.fitted_params = {param: np.full(len(self), np.nan) for param in self.msd_model.params}

            self._calc_beta()
            self._calc_g1_norm()

            for i in range(len(self)):
                # Fit the MSD params and store the results in fitted_params
                curr_params = self._fit_msd_params(i)
                for param in curr_params:
                    self.fitted_params[param][i] = curr_params[param]
        elif self.beta_calculator.mode == "fit":
            pass

    def _calc_beta(self):
        """
        Calculates the beta parameter based on the beta_calculator attribute and stores it in the beta attribute.
        Only called if beta_calculator.mode is "fixed" or "raw".

        The beta attribute is a vector of the same length as the number of measurements, where each element is the beta
        parameter for the corresponding measurement.

        :return: None
        """
        if self.beta_calculator.mode == "fixed":
            self.beta = np.full((len(self),), self.beta_calculator.beta_fixed)
        elif self.beta_calculator.mode == "raw":
            idx_first = np.argmax(self.tau > self.beta_calculator.tau_lims[0])  # First index after the lower limit
            idx_last = np.argmax(self.tau > self.beta_calculator.tau_lims[1])  # First index after the upper limit
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims should be greater than the lower limit")
            self.beta = np.mean(self.g2_norm[idx_first:idx_last, :], axis=0) - 1

    def _calc_g1_norm(self):
        """
        Calculates the normalized first-order autocorrelation function g1_norm for all measurements using the
        previously calculated beta values and stores it in the g1_norm attribute.

        :return: None
        """
        self.g1_norm = np.sqrt(np.abs((self.g2_norm - 1) / self.beta))

    def _fit_msd_params(self, i: int) -> Dict:
        """
        Fits only the MSD params of the model (i.e., no beta) to a single measurement.

        :param i: Index of the measurement to fit.
        :return: A Dict with the fitted parameters.
        """
        # Crop the data based on tau_lims_fit and g1_lim_fit
        tau_fit, g1_norm_fit = self._crop_to_fit_interval(i)

        # Scale factor for the parameters. The parameters are scaled so that they are around 1, which helps the
        # optimization algorithm.
        scale_array = np.fromiter(self.msd_model.params_scale.values(), dtype=float)

        # Define the objective function for the fit
        def objective(params: np.ndarray) -> float:
            """
            Objective function for the fit.
            :param params: A vector of the parameters of the model to fit. The order of the parameters should match
                the order of the params attribute of the msd_model.
            :return: The sum of the squared differences between the model and the data.
            """
            params *= scale_array  # Scale the parameters back to their original values
            msd = self.msd_model.msd_fn(tau_fit, *params)
            g1_norm = hsi.g1_norm(msd, self.mua[i], self.musp[i], self.rho, self.n, self.lambda0)
            return np.sum((g1_norm - g1_norm_fit) ** 2)

        # Perform the fit
        scaled_x0 = np.fromiter(self.msd_model.param_init.values(), dtype=float) / scale_array
        bounds = list(self.msd_model.param_bounds.values())
        # Scale the bounds, but set to None if the original bound is None
        scaled_bounds = [(None if bound[0] is None else bound[0] / scale_array[i],
                          None if bound[1] is None else bound[1] / scale_array[i]) for i, bound in enumerate(bounds)]
        res = opt.minimize(
            fun=objective,
            x0=scaled_x0,
            bounds=scaled_bounds
        )

        if not res.success:
            fitted_params = dict(zip(self.msd_model.params, np.full(len(res.x), np.nan)))
        else:
            fitted_params = dict(zip(self.msd_model.params, res.x * scale_array))

        return fitted_params

    def _crop_to_fit_interval(self, i: int) -> tuple:
        """
        Crops the data based on tau_lims_fit and g1_lim_fit.
        :param i: Index of the measurement to crop.
        :return: The cropped tau and g1_norm.
        """
        if self.tau_lims_fit is not None and self.g1_lim_fit is not None:
            idx_first = np.argmax(self.tau > self.tau_lims_fit[0])  # First index after the lower limit
            idx_last_tau = np.argmax(self.tau > self.tau_lims_fit[1])  # First index after the upper limit
            idx_last_g1 = np.argmax(self.g1_norm[:, i] < self.g1_lim_fit)  # First index after g1_lim_fit
            idx_last = min(idx_last_tau, idx_last_g1)
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims_fit should be greater than the lower limit or the first "
                                 "g1_norm value greater than g1_lim_fit")
            tau_fit = self.tau[idx_first:idx_last]
            g1_norm_fit = self.g1_norm[idx_first:idx_last, i]
        elif self.tau_lims_fit is not None:
            idx_first = np.argmax(self.tau > self.tau_lims_fit[0])  # First index after the lower limit
            idx_last = np.argmax(self.tau > self.tau_lims_fit[1])  # First index after the upper limit
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims_fit should be greater than the lower limit")
            tau_fit = self.tau[idx_first:idx_last]
            g1_norm_fit = self.g1_norm[idx_first:idx_last, i]
        elif self.g1_lim_fit is not None:
            idx_last = np.argmax(self.g1_norm[:, i] < self.g1_lim_fit)  # First index after g1_lim_fit
            tau_fit = self.tau[:idx_last]
            g1_norm_fit = self.g1_norm[:idx_last, i]
        else:
            tau_fit = self.tau
            g1_norm_fit = self.g1_norm[:, i]

        return tau_fit, g1_norm_fit
