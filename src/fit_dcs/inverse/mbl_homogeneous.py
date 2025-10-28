import numpy as np
import fit_dcs.forward.common as common
from typing import Callable, Literal


class MSDModelMBL:
    """
    A class for specifying the mean-square displacement (MSD) model for analyzing the normalized second-order
    autocorrelation functions g2_norm using the Modified Beer-Lambert law. The possible models are:

    - "brownian": Brownian motion model with a single parameter, the diffusion coefficient (db).

    - "ballistic": ballistic motion model with a single parameter, the mean square speed of the
        scatterers (v_ms).
    """

    def __init__(self, model_name: Literal["brownian", "ballistic"], param0: float):
        """
        Class constructor.

        :param model_name: The name of the model to use for the mean-square displacement.
            Choose between "brownian" or "ballistic".
        :param param0: The baseline value of the parameter, typically estimated previously via a fit. If model is
            "brownian", this is the baseline diffusion coefficient db [cm^2/s]. If model is "ballistic", this is the
            baseline mean square speed v_ms [cm/s].
        """
        # Check the model and fetch the appropriate function and derivative. Also check that the baseline parameters
        # are provided.
        if model_name == "brownian":
            self.msd_fn = common.msd_brownian
            self.d_msd_fn = common.d_msd_brownian
        elif model_name == "ballistic":
            self.msd_fn = common.msd_ballistic
            self.d_msd_fn = common.d_msd_ballistic
        else:
            raise ValueError(f"Model {model_name} not recognized. Choose between 'brownian' or 'ballistic'.")
        self.model = model_name
        self.param0 = param0


class MBLHomogeneous:
    """
    Class for extracting Db from DCS data using the DCS Modified Beer-Lambert law for a homogeneous semi-infinite
    medium.
    """

    def __init__(
            self,
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
            different iteration, and each row corresponds to a different time delay. The number of rows should be the
            same as the length of tau.
        :param g2_norm_0: Normalized second-order autocorrelation function for the baseline measurement. A vector the
            same length as tau.
        :param d_factors_fn: The function to use to calculate the d factors for the DCS Modified Beer-Lambert law. This
            typically comes from the forward module.
        :param msd_model: An instance of the MSDModelMBL class that specifies the mean-square displacement model to use.
        :param mua: Absorption coefficient of the medium [1/cm]. If a float, the same value is used for all
            iterations. If an array, a different value is used for each iteration, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param musp: Reduced scattering coefficient of the medium [1/cm]. If a float, the same value is used for all
            iterations. If an array, a different value is used for each iteration, and the length of the array
            should be the same as the number of columns in g2_norm.
        :param kwargs: Arguments to be passed to d_factors_fn, which gets called as d_factors_fn(msd0, **kwargs), where
            msd0 gets calculated using the baseline parameter provided in msd_model.
        """
        self.g2_norm_0 = g2_norm_0
        self.mua = mua
        self.musp = musp
        self.d_factors_fn = d_factors_fn
        self.msd_model = msd_model
        self.d_factors_fn_args = kwargs

    def fit(self, tau: np.ndarray, g2_norm: np.ndarray) -> np.ndarray:
        """
        Uses the DCS Modified Beer-Lambert law to calculate the msd parameter (Brownian diffusion coefficient or mean
        square velocity, depending on the model) for each lag time and iteration.

        :param tau: Vector of time delays [s].
        :param g2_norm: 2D array of normalized second-order autocorrelation functions. First dimension is iterations,
            second dimension is time delays.
        :return: The calculated parameter for each lag time and iteration. A matrix the same shape as g2_norm.
        """
        # Calculate variations in mua and musp from the baseline
        delta_mua = self.mua - self.d_factors_fn_args["mua0"]
        delta_musp = self.musp - self.d_factors_fn_args["musp0"]

        # Calculate the d factors for the baseline and broadcast them to the same shape as g2_norm
        msd0 = self.msd_model.msd_fn(tau, self.msd_model.param0)
        (dr, da, ds) = self.d_factors_fn(msd0, **self.d_factors_fn_args)
        # Calculate the d factor with respect to the parameter of interest based on dr and the derivative of the MSD
        dp = dr * self.msd_model.d_msd_fn(tau)
        # Broadcast the d factors to the same shape as g2_norm
        dp = np.expand_dims(dp, axis=0)
        dp = np.broadcast_to(dp, g2_norm.shape)
        da = np.expand_dims(da, axis=0)
        da = np.broadcast_to(da, g2_norm.shape)
        ds = np.expand_dims(ds, axis=0)
        ds = np.broadcast_to(ds, g2_norm.shape)

        # Calculate parameter of interest (Db or v_ms) for each iteration and lag time.
        g2_norm_0 = np.expand_dims(self.g2_norm_0, axis=0)
        g2_norm_0 = np.broadcast_to(g2_norm_0, g2_norm.shape)
        delta_od = -np.log((g2_norm - 1) / (g2_norm_0 - 1))
        delta_param = (delta_od - da * delta_mua - ds * delta_musp) / dp
        param = self.msd_model.param0 + delta_param

        return param


if __name__ == "__main__":
    import fit_dcs.forward.homogeneous_semi_inf as hsi

    tau = np.logspace(-7, -2, 200)
    db_true = 5e-8
    mua = 0.1
    musp = 10
    n = 1.4
    rho = 2.5
    lambda0 = 785
    beta_true = 0.5
    msd_true = common.msd_brownian(tau, db_true)
    g1_norm_true = hsi.g1_norm(msd_true, mua, musp, n, rho, lambda0)
    g2_norm_true = 1 + beta_true * g1_norm_true ** 2

    # Add fake noise to the g2_norm curve
    noise_level = 0.02
    n_samples = 5
    g2_norm_noisy = np.array([np.random.normal(g2_norm_true, noise_level, size=g2_norm_true.shape)
                              for _ in range(n_samples)])

    msd_model = MSDModelMBL(model_name="brownian", param0=db_true)
    fitter = MBLHomogeneous(
        g2_norm_0=g2_norm_true,
        d_factors_fn=hsi.d_factors,
        msd_model=msd_model,
        mua=mua,
        musp=musp,
        mua0=mua,
        musp0=musp,
        rho=rho,
        n=n,
        lambda0=lambda0
    )
    db_fit = fitter.fit(tau, g2_norm_noisy)
    print(np.nanmedian(db_fit, axis=1))
