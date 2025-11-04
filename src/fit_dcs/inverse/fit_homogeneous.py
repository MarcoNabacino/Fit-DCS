from typing import Dict, Callable, Literal
import numpy as np
import fit_dcs.forward.common as common
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import warnings


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

    def __init__(self, model_name: Literal["brownian", "ballistic", "hybrid"], param_init: Dict,
                 param_bounds: Dict | None = None):
        """
        Class constructor.

        :param model_name: The name of the MSD model. Choose from "brownian", "ballistic", or "hybrid".
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

    - "fit": The beta parameter is fitted along with the MSD parameters using the same time interval as the MSD
        parameters. The initial value of beta is defined by beta_init, and the bounds are defined by beta_bounds.
    """

    def __init__(self, mode: Literal["fixed", "raw", "fit"], **kwargs):
        """
        Class constructor.

        :param mode: The mode of the beta calculation. Either "fixed", "raw", or "fit".
        :param kwargs: Additional arguments depending on the mode. Specific arguments are:
            - If mode is "fixed", then beta_fixed should be a float.
            - If mode is "raw", then tau_lims should be an ordered pair of floats.
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
                    # Warn the user if beta_fixed is outside the range [0, 1]
                    if self.beta_fixed < 0 or self.beta_fixed > 1:
                        print("Warning: beta_fixed should be in the range [0, 1]")
                else:
                    raise ValueError("beta_fixed should be a float")
            else:
                raise ValueError("beta_fixed should be provided for the 'fixed' mode")
        elif mode == "raw":
            if "tau_lims" in kwargs:
                # Check that tau_lims is an ordered pair of floats
                if isinstance(kwargs["tau_lims"], (tuple, list)) and len(kwargs["tau_lims"]) == 2:
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
                # If beta_bounds is a 2-tuple, check that the elements are floats or None.
                if isinstance(kwargs["beta_bounds"], (tuple, list)) and len(kwargs["beta_bounds"]) == 2:
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
            raise ValueError(f"Unknown mode: {mode}. Choose from 'fixed', 'raw', or 'fit'")

        self.mode = mode


class FitHomogeneous:
    """
    A class for fitting the normalized second-order autocorrelation functions g2_norm using a homogeneous model for an
    arbitrary geometry (e.g., semi-infinite, laterally infinite slab, etc.) and a mean-square displacement (MSD) model
    (e.g., Brownian, ballistic, hybrid).
    """

    def __init__(
            self,
            g1_norm_fn: Callable,
            msd_model: MSDModelFit,
            beta_calculator: BetaCalculator,
            tau_lims_fit: tuple[float, float] | None = None,
            g2_lim_fit: float | None = None,
            **g1_args
    ):
        """
        Class constructor.

        :param g1_norm_fn: A function that calculates the normalized first-order autocorrelation function g1_norm
            based on the MSD model. This should be one of the functions defined in the forward module. The function
            has the signature g1_norm_model(msd, **g1_args), where msd is the mean-square displacement of the
            scatterers, and **g1_args are the additional arguments needed for the function.
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
            used for all iterations. If fitting multiple curves simultaneously (e.g., multiple source-detector
            separations), then the keyword arguments can also be tuples, where each element corresponds to a different
            curve and can be either a float or a vector of floats as described before.
        """
        # Check that tau_lims_fit is an ordered pair of floats
        if tau_lims_fit is not None:
            if tau_lims_fit[0] < tau_lims_fit[1]:
                self.tau_lims_fit = tau_lims_fit
            else:
                raise ValueError(f"tau_lims_fit should be an ordered pair of floats, got {tau_lims_fit}")
        else:
            self.tau_lims_fit = None

        self.g1_norm_fn = g1_norm_fn

        # Get number of multi-curves from g1_args. Additionally, check that, for g1_args that are tuples,
        # all tuples have the same length
        self.n_multi = 1
        for key, value in g1_args.items():
            if isinstance(value, tuple):
                if self.n_multi == 1:
                    self.n_multi = len(value)
                elif len(value) != self.n_multi:
                    raise ValueError(f"All g1_args tuples should have the same length")
        # Now that n_multi is known, convert all g1_args to tuples of length n_multi
        for key, value in g1_args.items():
            if not isinstance(value, tuple):
                g1_args[key] = self.n_multi * (value,)

        self.g1_args = g1_args
        self.msd_model = msd_model
        self.beta_calculator = beta_calculator
        self.g2_lim_fit = g2_lim_fit

        # Initialize the beta, fitted_params, and chi2 attributes.
        self.beta = None
        self.msd_params = None
        self.chi2 = None
        self.r2 = None

    def fit(self, tau, g2_norm, plot_interval: int = 0) -> pd.DataFrame:
        """
        Fits the model to the data and returns the results as a DataFrame.

        :param tau: Vector of time delays [s].
        :param g2_norm: 2D or 3D array of normalized second-order autocorrelation functions. First dimension is
            iterations, second dimension is multi-curves (if any), third dimension is time delays.
        :param plot_interval: If not 0, a plot showing the g2_norm curves and the fitted curves is displayed every
            plot_interval iterations. Default is 0 (no plots).
        :return: A DataFrame with the fitted MSD parameters (column names: 'db' and/or 'v_ms' ), 'beta', 'chi2',
            and 'r2' for each iteration.
        """
        # Expand g2_norm to 3D if it is 2D and check shape consistency
        if g2_norm.ndim == 2:
            g2_norm = g2_norm[:, np.newaxis, :]
        n_iter = g2_norm.shape[0]
        if g2_norm.shape[1] != self.n_multi:
            raise ValueError("Shape mismatch: g2_norm second dimension should match the number of multi-curves "
                             f"defined in g1_args. Expected {self.n_multi}, got {g2_norm.shape[1]}")
        if g2_norm.shape[-1] != len(tau):
            raise ValueError("Shape mismatch: g2_norm should have the same number of columns as the length of tau")

        self.beta = np.full((n_iter, self.n_multi), np.nan)
        self.msd_params = {param: np.full(n_iter, np.nan) for param in self.msd_model.params}
        self.chi2 = np.full(n_iter, np.nan)
        self.r2 = np.full(n_iter, np.nan)

        if self.beta_calculator.mode in ["fixed", "raw"]:
            self.beta = self._calc_beta(tau, g2_norm)

        for i in range(n_iter):
            curr_params, self.beta[i, :], self.chi2[i], self.r2[i] = self._process_iteration(i, tau, g2_norm)
            for param in curr_params:
                self.msd_params[param][i] = curr_params[param]

            if plot_interval > 0 and i % plot_interval == 0:
                f = self._plot_fit(i, tau, g2_norm)
                f.show()

       # Create a DataFrame with the fitted parameters and beta
        df = pd.DataFrame(self.msd_params)
        # Create 1 column per multi-curve to save beta
        if self.n_multi == 1:
            df["beta"] = np.squeeze(self.beta)
        else:
            for j in range(self.n_multi):
                df[f"beta_{j}"] = self.beta[:, j]
        df["chi2"] = self.chi2
        df["r2"] = self.r2

        return df

    def _calc_beta(self, tau, g2_norm):
        """
        Calculates the beta parameter for all iterations based on the beta_calculator attribute.
        Only called if beta_calculator.mode is "fixed" or "raw".

        :param tau: Vector of time delays [s].
        :param g2_norm: 2D array of normalized second-order autocorrelation functions. First dimension is time delays,
            second dimension is iterations.
        :return: The calculated beta parameter as a 2D array of shape (n_iter, n_multi).
        """
        n_iter = g2_norm.shape[0]
        if self.beta_calculator.mode == "fixed":
            return np.full((n_iter, self.n_multi), self.beta_calculator.beta_fixed)

        if self.beta_calculator.mode == "raw":
            idx_first = np.argmax(tau > self.beta_calculator.tau_lims[0])  # First index after the lower limit
            idx_last = np.argmax(tau > self.beta_calculator.tau_lims[1])  # First index after the upper limit
            if idx_first >= idx_last:
                raise ValueError("The upper limit of tau_lims should be greater than the lower limit")
            return np.mean(g2_norm[:, :, idx_first:idx_last], axis=-1) - 1  # Shape (n_iter, n_multi)

        raise ValueError(f"Unknown beta calculation mode: {self.beta_calculator.mode}")

    def _process_iteration(self, i: int, tau: np.ndarray, g2_norm: np.ndarray) -> tuple[Dict, float, float, float]:
        """
        Fits only the MSD params of the model (i.e., no beta) to a single iteration.
        Processes a single iteration (i.e., a single g2_norm), returning the fitted MSD parameters, beta, chi2 and r2.

        :param i: Index of the iteration to fit.
        :param tau: Vector of time delays [s].
        :param g2_norm: 3D array of normalized second-order autocorrelation functions. First dimension is iterations,
            second dimension is multi-curve, third dimension is time delays.
        :return: A 4-tuple containing: (a Dict with the fitted msd parameters, beta, chi2, r2). beta is a 1D array
            with the beta values for each multi-curve.
        """
        # Get the boolean mask to crop g2_norm and tau for fitting
        crop_mask = self._get_crop_mask(tau, g2_norm[i, :, :])

        # Scale factor for the parameters. The parameters are scaled so that they are around 1, which helps the
        # optimization algorithm.
        scale_array = np.fromiter(self.msd_model.params_scale.values(), dtype=float)

        scaled_msd_params_init = np.fromiter(self.msd_model.param_init.values(), dtype=float) / scale_array
        if self.beta_calculator.mode == "fit":
            beta_init = np.full(self.n_multi, self.beta_calculator.beta_init)
            scaled_x0 = np.concatenate((scaled_msd_params_init, beta_init))
        else:
            scaled_x0 = scaled_msd_params_init
        msd_bounds = list(self.msd_model.param_bounds.values())
        scaled_msd_bounds = [
            (lo / s if lo is not None else None,
             hi / s if hi is not None else None)
            for (lo, hi), s in zip(msd_bounds, scale_array)
        ]
        if self.beta_calculator.mode == "fit":
            scaled_beta_bounds = self.n_multi * [self.beta_calculator.beta_bounds]
            scaled_bounds = scaled_msd_bounds + scaled_beta_bounds
        else:
            scaled_bounds = scaled_msd_bounds

        # Define the objective function for the fit
        def objective(params: np.ndarray) -> float:
            """
            Objective function for the fit.
            :param params: A vector of the parameters of the model to fit. The order of the parameters should match
                the order of the params attribute of the msd_model, followed by the beta parameter.
            :return: The sum of the squared differences between the model and the data.
            """
            if self.beta_calculator.mode == "fit":
                msd_params = params[:-self.n_multi]
                beta_multi = params[-self.n_multi:]
            else:
                msd_params = params
                beta_multi = self.beta[i, :]
            msd_params *= scale_array  # Scale the parameters back to their original values
            sse_total = 0.0
            for j in range(self.n_multi):
                mask_j = crop_mask[j, :]
                if not np.any(mask_j):
                    continue
                tau_j = tau[mask_j]
                msd_j = self.msd_model.msd_fn(tau_j, *msd_params)
                g1_args_j = self._get_g1_args(i, j)
                g1_norm_j = self.g1_norm_fn(msd_j, **g1_args_j)

                beta_j = beta_multi[j]
                g2_norm_model_j = 1 + beta_j * g1_norm_j ** 2

                g2_norm_j = g2_norm[i, j, :][mask_j]
                sse_total += np.sum((g2_norm_model_j - g2_norm_j) ** 2)

            return sse_total

        # Perform the fit
        res = opt.minimize(
            fun=objective,
            x0=scaled_x0,
            bounds=scaled_bounds
        )

        if res.success:
            if self.beta_calculator.mode == "fit":
                msd_params = res.x[:-self.n_multi] * scale_array
                beta = res.x[-self.n_multi:]
            else:
                msd_params = res.x * scale_array
                beta = self.beta[i, :]
            msd_params_dict = dict(zip(self.msd_model.params, msd_params))

            # Calculate chi2 and r2 by averaging over all multi-curves
            chi2 = 0.0
            r2 = 0.0
            n_multi_valid = self.n_multi  # Used to average over valid multi-curves only
            for j in range(self.n_multi):
                mask_j = crop_mask[j, :]
                if not np.any(mask_j):
                    # No valid data points for this multi-curve, skip it
                    n_multi_valid -= 1
                    continue
                tau_j = tau[mask_j]
                g2_norm_cropped = g2_norm[i, j, :][mask_j]

                msd_j = self.msd_model.msd_fn(tau_j, *msd_params)

                g1_args_j = self._get_g1_args(i, j)

                g1_norm_j = self.g1_norm_fn(msd_j, **g1_args_j)
                g2_norm_best_fit = 1 + beta[j] * g1_norm_j ** 2

                square_err = (g2_norm_cropped - g2_norm_best_fit) ** 2
                chi2 += np.sum(square_err / g2_norm_best_fit)
                r2 += 1 - np.sum(square_err) / np.sum((g2_norm_cropped - np.mean(g2_norm_cropped)) ** 2)
            chi2 /= n_multi_valid
            r2 /= n_multi_valid
        else:
            msd_params_dict = dict(zip(self.msd_model.params, np.full(len(res.x), np.nan)))
            beta = np.full(self.n_multi, np.nan) if self.beta_calculator.mode == "fit" else self.beta[i, :]
            chi2 = np.nan
            r2 = np.nan

        return msd_params_dict, beta, chi2, r2

    def _get_crop_mask(self, tau: np.ndarray, g2_norm_multi: np.ndarray) -> np.ndarray:
        """
        Finds the boolean mask to apply to g2_norm_multi based on tau_lims_fit and g2_lim_fit.

        :param tau: Vector of time delays [s].
        :param g2_norm_multi: 2D vector of a multi-curve normalized second-order autocorrelation function.
            First dimension is multi-curve, second dimension is time delays.
        :return: A 2D boolean mask with the same shape as g2_norm_multi, where True indicates the values
            to keep for fitting.
        """
        n_tau = len(tau)
        mask = np.ones((self.n_multi, n_tau), dtype=bool)

        # Default full range
        if self.tau_lims_fit is None and self.g2_lim_fit is None:
            return mask

        # Tau limits (same for all multi-curves)
        if self.tau_lims_fit is not None:
            idx_first = np.searchsorted(tau, self.tau_lims_fit[0], side="right")
            idx_last_tau = np.searchsorted(tau, self.tau_lims_fit[1], side="right")
            if idx_first >= idx_last_tau:
                warnings.warn(f"Last index ({idx_last_tau}) <= first index ({idx_first}). "
                              f"Falling back to using the entire tau range")
                idx_first, idx_last_tau = 0, n_tau
        else:
            idx_first, idx_last_tau = 0, n_tau

        # g2 limits (can be different for each multi-curve)
        if self.g2_lim_fit is not None:
            # For each multi-curve, find the last index where g2_norm_multi > g2_lim_fit
            greater = g2_norm_multi > self.g2_lim_fit  # Shape (n_multi, n_tau)
            idx_last_g2 = np.full(self.n_multi, n_tau, dtype=int)
            for j in range(self.n_multi):
                inds = np.nonzero(greater[j])[0]
                if inds.size > 0:
                    idx_last_g2[j] = inds[-1] + 1  # Exclusive index
                else:
                    # No values greater than g2_lim_fit, use full range
                    warnings.warn(f"Curve {j}: no g2_norm values greater than g2_lim_fit ({self.g2_lim_fit}). "
                                  f"Using limits defined by tau_lims_fit only.")
                    idx_last_g2[j] = n_tau
        else:
            idx_last_g2 = np.full(self.n_multi, n_tau, dtype=int)

        # For each curve, the last index to use is the minimum between idx_last_tau and idx_last_g2
        idx_first_arr = np.full(self.n_multi, idx_first, dtype=int)
        idx_last_arr = np.minimum(idx_last_tau, idx_last_g2)

        # If for any curves idx_first >= idx_last, fall back to using the entire tau range for that curve
        for j in range(self.n_multi):
            if idx_first_arr[j] >= idx_last_arr[j]:
                warnings.warn(f"Curve {j}: last index ({idx_last_arr[j]}) <= first index ({idx_first_arr[j]}). "
                              f"Falling back to using the entire tau range for that curve")
                idx_first_arr[j], idx_last_arr[j] = 0, n_tau

        # Build mask
        for j in range(self.n_multi):
            mask[j, :idx_first_arr[j]] = False
            mask[j, idx_last_arr[j]:] = False

        return mask

    def _plot_fit(self, i: int, tau: np.ndarray, g2_norm: np.ndarray) -> plt.Figure:
        """
        Plots the g2_norm curve and the fitted curve for a single iteration, also displaying the fitting interval.
        Displays 1 subplot per multi-curve.

        :param i: The index of the iteration to plot
        :param tau: Vector of time delays [s].
        :param g2_norm: 2D array of normalized second-order autocorrelation functions. First dimension is iterations,
            second dimension is time delays.
        :return: The figure object
        """
        mask = self._get_crop_mask(tau, g2_norm[i, :, :])
        g2_norm_fit, msd_params, beta = self._get_best_fit(i, tau)

        fig, axs = plt.subplots(self.n_multi, 1, figsize=(6, 3 * self.n_multi), sharex=True)
        ax = None
        if self.n_multi == 1:
            axs = [axs]
        for j in range(self.n_multi):
            ax = axs[j]
            tau_cropped = tau[mask[j, :]]
            g2_norm_j = g2_norm[i, j, :]
            g2_norm_fit_cropped = g2_norm_fit[j, :][mask[j, :]]
            ax.semilogx(tau, g2_norm_j, marker='.', linestyle='none', label="Data")
            ax.semilogx(tau_cropped, g2_norm_fit_cropped, label="Fit")
            text = rf"$\beta$ = {beta[j]:.2f}"
            ax.annotate(text, xy=(0.05, 0.20), xycoords="axes fraction", fontsize=10, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            ax.set_ylabel(r"$g^{(2)}(\tau)$")
            ax.legend()
        ax.set_xlabel(r"$\tau$ [s]")
        fig.suptitle(f"Iteration {i}\n"
                     f"{' ,'.join([f'{param}={msd_params[param]:.2e}' for param in msd_params])}")

        return fig

    def _get_best_fit(self, i: int, tau: np.ndarray) -> tuple[np.ndarray, Dict, np.ndarray]:
        """
        Calculates the best fit g2_norm curve for all multi-curves of a single iteration using the fitted parameters.
        To be called after fitting.

        :param i: The index of the iteration to calculate the best fit g2_norm curve.
        :param tau: Vector of time delays [s].
        :return: A tuple containing:
            - A 2D array with shape (n_multi, n_tau) with the best fit g2_norm curves for all multi-curves.
            - A Dict with the fitted MSD parameters.
            - A 1D array with the beta values for each multi-curve.
        """
        msd_params = {param: self.msd_params[param][i] for param in self.msd_model.params}
        beta = self.beta[i, :]

        msd = self.msd_model.msd_fn(tau, *msd_params.values())

        g2_norm = np.empty((self.n_multi, len(tau)))
        for j in range(self.n_multi):
            g1_args_j = self._get_g1_args(i, j)
            g1_norm_j = self.g1_norm_fn(msd, **g1_args_j)
            g2_norm[j, :] = 1 + beta[j] * g1_norm_j ** 2

        return g2_norm, msd_params, beta

    def _get_g1_args(self, i: int, j: int) -> Dict:
        """
        Retrieves the g1_args for the i-th iteration and j-th multi-curve.

        :param i: Index of the iteration.
        :param j: Index of the multi-curve.
        :return: A dictionary with the g1_args for the specified iteration and multi-curve.
        """
        g1_args_j = {}
        for key, arg_tuple in self.g1_args.items():
            elem = arg_tuple[j]
            if isinstance(elem, (np.ndarray, list)):
                g1_args_j[key] = elem[i]
            else:
                g1_args_j[key] = elem
        return g1_args_j


if __name__ == "__main__":
    import fit_dcs.forward.homogeneous_semi_inf as hsi

    tau = np.logspace(-7, -2, 200)
    db_true = 5e-8
    mua = 0.1
    musp = 10
    n = 1.4
    rho = (1.5, 2.5)
    lambda0 = 785
    beta_true = 0.5
    msd_true = common.msd_brownian(tau, db_true)
    n_iter = 5
    g1_norm_true = np.empty((n_iter, len(rho), len(tau)))
    for j in range(len(rho)):
        g1_norm_true[:, j, :] = hsi.g1_norm(msd_true, mua, musp, rho[j], n, lambda0)
    g2_norm_true = 1 + beta_true * g1_norm_true ** 2

    # Add fake noise to the g2_norm curve
    noise_level = 0.02
    g2_norm_noisy = np.random.normal(g2_norm_true, noise_level)

    msd_model = MSDModelFit(model_name="brownian", param_init={"db": 1e-8},
                            param_bounds={"db": (0, None)})
    beta_calculator = BetaCalculator(mode="fit", beta_init=0.48, beta_bounds=(0, 1))
    fitter = FitHomogeneous(
        g1_norm_fn=hsi.g1_norm,
        msd_model=msd_model,
        beta_calculator=beta_calculator,
        tau_lims_fit=(1e-7, 1e-2),
        g2_lim_fit=1.13,
        mua=mua,
        musp=musp,
        n=n,
        rho=rho,
        lambda0=lambda0
    )
    results_df = fitter.fit(tau, g2_norm_noisy, plot_interval=1)
    print(results_df)