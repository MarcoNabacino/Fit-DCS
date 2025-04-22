from typing import Dict, Callable
import numpy as np
import forward.common as common
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd


class MSDModelFit:
    """
    A class for specifying the mean-square displacement (MSD) model for fitting the normalized second-order
    autocorrelation functions g2_norm. The possible models are:

    - "brownian": Brownian motion model with a single parameter, the diffusion coefficient (db).

    - "ballistic": ballistic motion model with a single parameter, the mean square speed of the
        scatterers (v_ms).

    - "hybrid": hybrid model that combines Brownian and ballistic motion with two parameters, the
        diffusion coefficient (db) and the mean square speed of the scatterers (v_ms).
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


class FitHomogeneous:
    """
    A class for fitting the normalized second-order autocorrelation functions g2_norm using a homogeneous model for an
    arbitrary geometry (e.g., semi-infinite, laterally infinite slab, etc.) and a mean-square displacement (MSD) model
    (e.g., Brownian, ballistic, hybrid).
    """

    def __init__(
            self,
            tau: np.ndarray,
            g2_norm: np.ndarray,
            g1_norm_fn: Callable,
            msd_model: MSDModelFit,
            beta_calculator: BetaCalculator,
            tau_lims_fit: tuple | None = None,
            g2_lim_fit: float | None = None,
            **kwargs
    ):
        """
        Class constructor.

        :param tau: Vector of time delays [s].
        :param g2_norm: Matrix of normalized second-order autocorrelation functions. Each column corresponds to a
            different iteration, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param g1_norm_fn: A function that calculates the normalized first-order autocorrelation function g1_norm
            based on the MSD model. This should be one of the functions defined in the forward module. The function
            has the signature g1_norm_model(msd, **kwargs), where msd is the mean-square displacement of the
            scatterers, and **kwargs are the additional arguments needed for the function.
        :param msd_model: An instance of the MSDModel class.
        :param beta_calculator: An instance of the BetaCalculator class.
        :param tau_lims_fit: Ordered pair of floats defining the lower and upper limits of the time delays used for
            fitting. If None, the entire tau range is used.
        :param g2_lim_fit: If not None, the portion of the g1_norm curve that is used for fitting is limited to the
            values greater than g2_lim_fit. If both tau_lims_fit and g1_lim_fit are provided, the fitting is done
            starting from tau_lims_fit[0] and up to the minimum of tau_lims_fit[1] and the first time delay where
            g2_norm is greater than g2_lim_fit.
        :param kwargs: Additional arguments needed for the g1_norm_model function (e.g., mua, musp, rho, etc.). Each
            keyword argument can either be a float or a vector of floats. If it is a vector, then the length of the
            vector should be the same as the number of iterations in g2_norm. If it is a float, then the same value is
            used for all iterations. The keyword arguments should be named the same as the arguments of the
            g1_norm_model function.
        """

        # Check that the number of rows in g2_norm is the same as the length of tau
        if g2_norm.shape[0] == len(tau):
            self.tau = tau
            self.g2_norm = g2_norm
        else:
            raise ValueError("The number of rows in g2_norm should be the same as the length of tau")

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

        self.g1_norm_fn = g1_norm_fn
        self.g1_args = kwargs
        self.msd_model = msd_model
        self.beta_calculator = beta_calculator
        self.g2_lim_fit = g2_lim_fit

        # Initialize the beta, fitted_params, and chi2 attributes.
        self.beta = np.full(len(self), np.nan)
        self.fitted_params = {param: np.full(len(self), np.nan) for param in self.msd_model.params}
        self.chi2 = np.full(len(self), np.nan)

    def __len__(self):
        """
        Returns the number of iterations.
        """
        return self.g2_norm.shape[1]

    def fit(self, plot_interval: int = 0) -> pd.DataFrame:
        """
        Fits the model to the data and stores the beta and fitted parameters in the beta and fitted_params attribute.

        :param plot_interval: If not 0, a plot showing the g2_norm curves and the fitted curves is displayed every
            plot_interval iterations. Default is 0 (no plots).
        :return: A DataFrame with beta and the fitted parameters for each iteration.
        """
        if self.beta_calculator.mode in ["fixed", "raw", "raw_weighted"]:
            self._calc_beta()

        for i in range(len(self)):
            if self.beta_calculator.mode in ["fixed", "raw", "raw_weighted"]:
                # Fit the MSD params and store the results in fitted_params
                curr_params, curr_chi2 = self._fit_msd_params(i)
            elif self.beta_calculator.mode == "fit":
                # Fit both the MSD params and beta and store the results in fitted_params and beta
                curr_params, curr_beta, curr_chi2 = self._fit_beta_and_msd_params(i)
                self.beta[i] = curr_beta

            for param in curr_params:
                self.fitted_params[param][i] = curr_params[param]

            self.chi2[i] = curr_chi2
            if plot_interval > 0 and i % plot_interval == 0:
                self._plot_fit(i)
                plt.show()

       # Create a DataFrame with the fitted parameters and beta
        df = pd.DataFrame(self.fitted_params)
        df["beta"] = self.beta
        df["chi2"] = self.chi2

        return df

    def _calc_beta(self):
        """
        Calculates the beta parameter for all iterations based on the beta_calculator attribute and stores it in the
        beta attribute. Only called if beta_calculator.mode is "fixed", "raw", or "raw_weighted".

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

    def _fit_msd_params(self, i: int) -> tuple[Dict, float]:
        """
        Fits only the MSD params of the model (i.e., no beta) to a single iteration.

        :param i: Index of the iteration to fit.
        :return: A 2-tuple containing a Dict with the fitted parameters and the chi2 value.
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
                the order of the params attribute of the msd_model.
            :return: The sum of the squared differences between the model and the data.
            """
            params *= scale_array  # Scale the parameters back to their original values
            msd = self.msd_model.msd_fn(tau_fit, *params)
            # self.g1_args contains the additional arguments needed for the g1_norm function, as vectors or floats.
            # We need to select the i-th element of each argument vector for the current iteration.
            g1_args = {key: value[i] if isinstance(value, np.ndarray) else value for key, value in self.g1_args.items()}
            g1_norm = self.g1_norm_fn(msd, **g1_args)
            g2_norm = 1 + self.beta[i] * g1_norm ** 2
            return np.sum((g2_norm - g2_norm_fit) ** 2)

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
            chi2 = np.nan
        else:
            fitted_params = dict(zip(self.msd_model.params, res.x * scale_array))
            chi2 = res.fun

        return fitted_params, chi2

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
            idx_last_g2 = indices[-1] # Last index where g2_norm > g2_lim_fit
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
        :return: A 3-tuple with the fitted beta, the fitted parameters' dictionary, and the chi2 value.
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
            # self.g1_args contains the additional arguments needed for the g1_norm function, as vectors or floats.
            # We need to select the i-th element of each argument vector for the current iteration.
            g1_args = {key: value[i] if isinstance(value, np.ndarray) else value for key, value in self.g1_args.items()}
            g1_norm = self.g1_norm_fn(msd, **g1_args)
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
            chi2 = np.nan
        else:
            fitted_params = dict(zip(self.msd_model.params, res.x[:-1] * scale_array))
            fitted_beta = res.x[-1]
            chi2 = res.fun

        return fitted_params, fitted_beta, chi2

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
        # self.g1_args contains the additional arguments needed for the g1_norm function, as vectors or floats.
        # We need to select the i-th element of each argument vector for the current iteration.
        g1_args = {key: value[i] if isinstance(value, np.ndarray) else value for key, value in self.g1_args.items()}
        g1_norm = self.g1_norm_fn(msd, **g1_args)
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