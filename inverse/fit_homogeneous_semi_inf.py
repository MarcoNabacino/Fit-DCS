from typing import Dict
import numpy as np
import forward.common as common
import forward.homogeneous_semi_inf as hsi
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import utils.noise as noise


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
        will be fit and their scale and offset values.
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
    Class defining how to calculate the beta parameter. The possible modes are:

    - "fixed": The beta parameter is fixed to a specific user-defined value (beta_fixed).

    - "raw": The beta parameter is calculated from the raw data as the mean of g2_norm - 1 over a specific time
        interval (defined by tau_lims).

    - "raw_weighted": beta is calculated from the raw data as the mean of (g2_norm - 1) * (1 + tau / tau_c) over a
        specific time interval (defined by tau_lims), where tau_c is estimated from the raw data using a simple
        exponential model. Specifically, alpha * tau_c is the time delay where g2_norm is equal to
        1 + beta0 * exp(-alpha).

    - "fit": The beta parameter is fitted along with the MSD parameters using the same time interval as the MSD
        parameters. The initial value of beta is defined by beta_init, and the bounds are defined by beta_bounds.
    """

    def __init__(self, mode: str, **kwargs):
        """
        Class constructor.

        :param mode: The mode of the beta calculation. Either "fixed", "raw", or "fit".
        :param kwargs: Additional arguments depending on the mode. Specific arguments are:
            - If mode is "fixed", then beta_fixed should be a float.
            - If mode is "raw", then tau_lims should be an ordered pair of floats.
            - If mode is "raw_weighted", then tau_lims should be an ordered pair of floats, beta0 should be a float,
                and alpha (optional, default 0.5) should be a float in the range [0, 1].
            - If mode is "fit", then beta_init should be a float and beta_bounds (optional) should be a 2-tuple
                specifying the lower and upper bounds of the beta parameter. Use None rather than inf for no bound.
                If beta_bounds is not provided, the default bounds are (None, None).
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
        elif mode == "raw_weighted":
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
                raise ValueError("tau_lims should be provided for the 'raw_weighted' mode")
            if "beta0" in kwargs:
                # Check that beta0 is a float
                if isinstance(kwargs["beta0"], (float, int)):
                    self.beta0 = kwargs["beta0"]
                    # Warn the user if beta0 is outside the range [0, 0.5]
                    if self.beta0 < 0 or self.beta0 > 0.5:
                        print("Warning: beta0 should be in the range [0, 0.5]")
                else:
                    raise ValueError("beta0 should be a float")
            else:
                raise ValueError("beta0 should be provided for the 'raw_weighted' mode")
            if "alpha" in kwargs:
                # Check that alpha is a float in the range [0, 1]
                if isinstance(kwargs["alpha"], (float, int)) and 0 <= kwargs["alpha"] <= 1:
                    self.alpha = kwargs["alpha"]
                else:
                    raise ValueError("alpha should be a float in the range [0, 1]")
            else:
                self.alpha = 0.5
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
                # If beta_bounds is a 2-tuple, check that the elements are floats or None.
                if isinstance(kwargs["beta_bounds"], tuple) and len(kwargs["beta_bounds"]) == 2:
                    if all(isinstance(x, (float, int, type(None))) for x in kwargs["beta_bounds"]):
                        # If both elements are floats, check that the bounds are in the correct order
                        if kwargs["beta_bounds"][0] is not None and kwargs["beta_bounds"][1] is not None:
                            if kwargs["beta_bounds"][0] >= kwargs["beta_bounds"][1]:
                                raise ValueError("The lower bound of beta_bounds should be less than the upper bound")
                        self.beta_bounds = kwargs["beta_bounds"]
                    else:
                        raise ValueError("beta_bounds should be a 2-tuple of floats or None")
                else:
                    raise ValueError("beta_bounds should be a 2-tuple")
            else:
                self.beta_bounds = (None, None)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'fixed', 'raw', 'raw_weighted', or 'fit'")

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
            g2_lim_fit: float | None = None,
            plot_interval: int | None = None,
            nfr: bool = False,
            time_integration: float | None = None,
            countrate: float | np.ndarray | None = None,
            n_speckle: int | None = None
    ):
        """
        Class constructor.

        :param tau: Vector of time delays [s].
        :param g2_norm: Matrix of normalized second-order autocorrelation functions. Each column corresponds to a
            different iteration, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param mua: Absorption coefficient of the medium [1/cm]. If a float, the same value is used for all
            iterations. If an array, a different value is used for each iteration, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param musp: Reduced scattering coefficient of the medium [1/cm]. If a float, the same value is used for all
            iterations. If an array, a different value is used for each iteration, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param rho: Source-detector separation [cm].
        :param n: Ratio of the refractive index of the medium to the refractive index of the surrounding medium
            (typically air).
        :param lambda0: Wavelength of the light source [nm].
        :param msd_model: An instance of the MSDModel class.
        :param beta_calculator: An instance of the BetaCalculator class.
        :param tau_lims_fit: Ordered pair of floats defining the lower and upper limits of the time delays used for
            fitting. If None, the entire tau range is used.
        :param g2_lim_fit: If not None, the portion of the g1_norm curve that is used for fitting is limited to the
            values greater than g2_lim_fit. If both tau_lims_fit and g1_lim_fit are provided, the fitting is done
            starting from tau_lims_fit[0] and up to the minimum of tau_lims_fit[1] and the first time delay where
            g2_norm is greater than g2_lim_fit.
        :param plot_interval: If not None, a plot showing the g2_norm curves and the fitted curves is displayed every
            plot_interval iterations.
        :param nfr: If True, the noise model is used during fitting, by minimizing |(g2_norm - g2_norm_model) / sigma|^2
            where sigma is the standard deviation of the noise. Default is False. If True, time_integration, countrate,
            and n_speckle should be provided. Note that NFR fitting is not supported for ballistic MSD models, or when
            beta_calculator.mode is "fit".
        :param time_integration: Integration time of the measurement [s]. Only needed for NFR fitting, ignored
            otherwise.
        :param countrate: The count rate of the measurement [Hz]. If a float, the same value is used for all iterations.
            If a vector, its length should be the same as the number of columns in g2_norm. Only needed for NFR fitting,
            ignored otherwise.
        :param n_speckle: Number of independent speckles that contribute to the measurement, that is, the number of
            curves that were averaged to obtain g2_norm. Only needed for NFR fitting, ignored otherwise.
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

        self.nfr = nfr
        # Check that, if nfr is True, then time_integration, countrate, and n_speckle are provided and valid.
        # Additionally, warn the user if the msd model is ballistic and if the beta_calculator mode is
        # "fit".
        if nfr:
            if time_integration is None:
                raise ValueError("time_integration should be provided when nfr is True")
            if countrate is None:
                raise ValueError("countrate should be provided when nfr is True")
            if n_speckle is None:
                raise ValueError("n_speckle should be provided when nfr is True")
            if isinstance(countrate, (float, int)):
                self.countrate = np.full(len(self), countrate)
            elif isinstance(countrate, np.ndarray):
                if len(countrate) == len(self):
                    self.countrate = countrate
                else:
                    raise ValueError("countrate should be a float or an array of the same length as the number of "
                                     "columns in g2_norm")
            else:
                raise ValueError("countrate should be a float or an array")
            self.n_speckle = n_speckle
            self.time_integration = time_integration
            if msd_model.model_name == "ballistic":
                print("Warning: NFR fitting is not supported for ballistic MSD models. Classical fitting will be "
                      "performed instead")
            if beta_calculator.mode == "fit":
                print("Warning: NFR fitting is not supported when beta_calculator.mode is 'fit'. Classical fitting "
                      "will be performed instead")

        self.rho = rho
        self.n = n
        self.lambda0 = lambda0
        self.msd_model = msd_model
        self.beta_calculator = beta_calculator
        self.g2_lim_fit = g2_lim_fit
        self.plot_interval = plot_interval if plot_interval is not None else 0

        # Initialize the beta and fitted_params attribute to None
        self.beta = None
        self.fitted_params = None

    def __len__(self):
        """
        Returns the number of iterations.
        """
        return self.g2_norm.shape[1]

    def fit(self) -> pd.DataFrame:
        """
        Fits the model to the data and stores the beta and fitted parameters in the beta and fitted_params attribute.
        :return: A DataFrame with beta and the fitted parameters for each iteration.
        """

        if self.beta_calculator.mode in ["fixed", "raw", "raw_weighted"]:
            # Initialize arrays to store the fitted parameters values
            self.fitted_params = {param: np.full(len(self), np.nan) for param in self.msd_model.params}

            self._calc_beta()
            self._calc_g1_norm()

            for i in range(len(self)):
                # Fit the MSD params and store the results in fitted_params
                curr_params = self._fit_msd_params(i)
                for param in curr_params:
                    self.fitted_params[param][i] = curr_params[param]

                if self.plot_interval > 0 and i % self.plot_interval == 0:
                    fig = self._plot_fit(i)
                    plt.show(fig)
        elif self.beta_calculator.mode == "fit":
            # Initialize arrays to store the fitted parameters values
            self.fitted_params = {param: np.full(len(self), np.nan) for param in self.msd_model.params}
            self.beta = np.full(len(self), np.nan)

            for i in range(len(self)):
                # Fit both the MSD params and beta and store the results in fitted_params and beta
                curr_params, curr_beta = self._fit_beta_and_msd_params(i)
                for param in curr_params:
                    self.fitted_params[param][i] = curr_params[param]
                self.beta[i] = curr_beta
                if self.plot_interval > 0 and i % self.plot_interval == 0:
                    fig = self._plot_fit(i)
                    plt.show(fig)

       # Create a DataFrame with the fitted parameters and beta
        df = pd.DataFrame(self.fitted_params)
        df["beta"] = self.beta

        return df

    def _calc_beta(self):
        """
        Calculates the beta parameter for all iterations based on the beta_calculator attribute and stores it in the
        beta attribute. Only called if beta_calculator.mode is "fixed" or "raw", or "raw_weighted".

        The beta attribute is a vector of the same length as the number of iterations, where each element is the beta
        parameter for the corresponding iteration.

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
        elif self.beta_calculator.mode == "raw_weighted":
            # Estimate tau_c using a simple exponential model
            g2_value = 1 + self.beta_calculator.beta0 * np.exp(-self.beta_calculator.alpha)
            # Find the last value of tau where g2_norm is greater than g2_value
            mask = self.g2_norm > g2_value
            last_indices = np.argmax(mask[::-1], axis=0) # Reverse search to find the last index in each iteration (column)
            last_indices = len(self.tau) - 1 - last_indices # Reverse the indices back
            tau_c = self.tau[last_indices] # Value of tau corresponding to the last index; this is alpha * tau_c
            tau_c /= self.beta_calculator.alpha # Divide by alpha to get tau_c
            # Calculate the weighted beta
            idx_first = np.argmax(self.tau > self.beta_calculator.tau_lims[0])  # First index after the lower limit
            idx_last = np.argmax(self.tau > self.beta_calculator.tau_lims[1])  # First index after the upper limit
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims should be greater than the lower limit")

            g2_norm_crop = self.g2_norm[idx_first:idx_last, :]
            tau_crop = self.tau[idx_first:idx_last]
            tau_crop = np.expand_dims(tau_crop, axis=-1)
            tau_crop = np.broadcast_to(tau_crop, g2_norm_crop.shape)
            self.beta = np.mean((self.g2_norm[idx_first:idx_last, :] - 1) * (1 + tau_crop / tau_c), axis=0)

    def _calc_g1_norm(self):
        """
        Calculates the normalized first-order autocorrelation function g1_norm for all iterations using the
        previously calculated beta values and stores it in the g1_norm attribute.

        :return: None
        """
        self.g1_norm = np.sqrt(np.abs((self.g2_norm - 1) / self.beta))

    def _fit_msd_params(self, i: int) -> Dict:
        """
        Fits only the MSD params of the model (i.e., no beta) to a single iteration.

        :param i: Index of the iteration to fit.
        :return: A Dict with the fitted parameters.
        """
        # Crop the data based on tau_lims_fit and g2_lim_fit
        idx_first, idx_last = self._crop_to_fit_interval(i)
        tau_fit = self.tau[idx_first:idx_last]
        g1_norm_fit = self.g1_norm[idx_first:idx_last, i]

        # Scale factor for the parameters. The parameters are scaled so that they are around 1, which helps the
        # optimization algorithm.
        scale_array = np.fromiter(self.msd_model.params_scale.values(), dtype=float)

        # Define the objective function for the fit
        if not self.nfr or self.msd_model.model_name == "ballistic":
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
        elif self.nfr and self.msd_model.model_name in ["brownian", "hybrid"]:
            # Fit the curve using a simple exponential model to find tau_c
            idx_last_simple_exp = np.argmax(self.tau > 1.56e-5)  # See documentation for utils.noise.NoiseAdder
            tau_simple_exp = self.tau[idx_first:idx_last_simple_exp]
            g2_simple_exp = self.g2_norm[idx_first:idx_last_simple_exp, i]
            tau_c = noise.fit_tau_c(self.beta[i], tau_simple_exp, g2_simple_exp)

            # Calculate the standard deviation of g2_norm based on the noise model
            sigma = noise.sigma_g2_norm(
                tau=tau_fit,
                t_integration=self.time_integration,
                countrate=self.countrate[i],
                beta=self.beta[i],
                tau_c=tau_c,
                n_speckle=self.n_speckle
            )

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
                return np.sum(((g1_norm - g1_norm_fit) / sigma) ** 2)

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
        Crops the data based on tau_lims_fit and g2_lim_fit.
        :param i: Index of the iteration to crop.
        :return: The indices of the first and last elements of the cropped data.
        """
        if self.tau_lims_fit is not None and self.g2_lim_fit is not None:
            idx_first = np.argmax(self.tau > self.tau_lims_fit[0])  # First index after the lower limit
            idx_last_tau = np.argmax(self.tau > self.tau_lims_fit[1])  # First index after the upper limit
            indices = np.where(self.g2_norm[:, i] > self.g2_lim_fit)[0] # Indices where g2_norm > g2_lim_fit
            idx_last_g2 = indices[-1] if len(indices) > 0 else -1 # Last index where g2_norm > g2_lim_fit
            idx_last = min(idx_last_tau, idx_last_g2)
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims_fit should be greater than the lower limit or the last "
                                 "g2_norm value smaller than g2_lim_fit")
        elif self.tau_lims_fit is not None:
            idx_first = np.argmax(self.tau > self.tau_lims_fit[0])  # First index after the lower limit
            idx_last = np.argmax(self.tau > self.tau_lims_fit[1])  # First index after the upper limit
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims_fit should be greater than the lower limit")
        elif self.g2_lim_fit is not None:
            idx_first = 0
            indices = np.where(self.g2_norm[:, i] > self.g2_lim_fit)[0] # Indices where g2_norm > g2_lim_fit
            idx_last = indices[-1] if len(indices) > 0 else -1 # Last index where g2_norm > g2_lim_fit
            if idx_first >= idx_last:
                raise ValueError("The last g2_norm value smaller than g2_lim_fit should exist")
        else:
            idx_first = 0
            idx_last = len(self.tau)

        return idx_first, idx_last

    def _fit_beta_and_msd_params(self, i: int) -> tuple:
        """
        Fits both the beta and MSD params of the model to a single iteration.

        :param i: Index of the iteration to fit.
        :return: A tuple with the fitted beta and the fitted parameters dictionary.
        """
        # Crop the data based on tau_lims_fit and g2_lim_fit
        idx_first, idx_last = self._crop_to_fit_interval(i)
        tau_fit = self.tau[idx_first:idx_last]
        g2_norm_fit = self.g2_norm[idx_first:idx_last, i]

        # Scale factor for the parameters. The parameters are scaled so that they are around 1, which helps the
        # optimization algorithm.
        scale_array = np.fromiter(self.msd_model.params_scale.values(), dtype=float)

        # Define the objective function for the fit
        def objective(params: np.ndarray) -> float:
            """
            Objective function for the fit.
            :param params: A vector of the parameters of the model to fit. The order of the parameters should match
                the order of the params attribute of the msd_model, followed by the beta parameter.
            :return: The sum of the squared differences between the model and the data.
            """
            # Split the parameters into MSD params and beta
            msd_params = params[:-1]
            beta = params[-1]
            msd_params *= scale_array
            msd = self.msd_model.msd_fn(tau_fit, *msd_params)
            g1_norm = hsi.g1_norm(msd, self.mua[i], self.musp[i], self.rho, self.n, self.lambda0)
            g2_norm = 1 + beta * g1_norm ** 2
            return np.sum((g2_norm - g2_norm_fit) ** 2)

        # Perform the fit
        scaled_x0 = np.concatenate((np.fromiter(self.msd_model.param_init.values(), dtype=float) / scale_array,
                                    [self.beta_calculator.beta_init]))
        bounds_msd = list(self.msd_model.param_bounds.values())
        bounds_beta = self.beta_calculator.beta_bounds
        # Scale the bounds, but set to None if the original bound is None
        scaled_bounds_msd = [(None if bound[0] is None else bound[0] / scale_array[i],
                              None if bound[1] is None else bound[1] / scale_array[i]) for i, bound in enumerate(bounds_msd)]
        scaled_bounds_beta = bounds_beta
        scaled_bounds = scaled_bounds_msd + [scaled_bounds_beta]
        res = opt.minimize(
            fun=objective,
            x0=scaled_x0,
            bounds=scaled_bounds
        )

        if not res.success:
            fitted_params = dict(zip(self.msd_model.params, np.full(len(res.x) - 1, np.nan)))
            fitted_beta = np.nan
        else:
            fitted_params = dict(zip(self.msd_model.params, res.x[:-1] * scale_array))
            fitted_beta = res.x[-1]

        return fitted_params, fitted_beta

    def _plot_fit(self, i: int) -> plt.figure:
        """
        Plots the g2_norm curve and the fitted curve for a single iteration, also displaying the fitting interval
        :param i: The index of the iteration to plot
        :return: The figure object
        """
        idx_first, idx_last = self._crop_to_fit_interval(i)
        tau_fit = self.tau[idx_first:idx_last]
        # Get the fitted params for iteration i
        fitted_params = {param: self.fitted_params[param][i] for param in self.msd_model.params}
        msd = self.msd_model.msd_fn(tau_fit, *fitted_params.values())
        g1_norm = hsi.g1_norm(msd, self.mua[i], self.musp[i], self.rho, self.n, self.lambda0)
        g2_norm = 1 + self.beta[i] * g1_norm ** 2

        f = plt.figure()
        plt.semilogx(self.tau, self.g2_norm[:, i], marker='.', linestyle='none', label="Data")
        plt.semilogx(tau_fit, g2_norm, label="Fit")
        text = f"β = {self.beta[i]:.2f}\n" + "\n".join([f"{k} = {v:.2e}" for k, v in fitted_params.items()])
        plt.annotate(text, xy=(0.05, 0.20), xycoords="axes fraction", fontsize=10,
                     verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        plt.title(f"Iteration {i}")
        plt.xlabel(r"$\tau$ [s]")
        plt.ylabel(r"$g^{(2)}(\tau)$")
        plt.legend()

        return f