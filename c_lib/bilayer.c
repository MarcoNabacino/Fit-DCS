#include <math.h>
#include <gsl/gsl_sf_bessel.h>

/*
 * Calculates the unnormalized G1 in the spatial frequency domain for a bilayer medium. From [1].
 *
 * [1] Wang, Q. et al. (2024). "A comprehensive overview of diffuse correlation spectroscopy: theoretical
 * framework, recent advances in hardware, analysis, and applications."
*/
double g1_spatial_freq(
    const double msd_up,  // Mean-square displacement in the upper layer [cm^2]
    const double mua_up,  // Absorption coefficient in the upper layer [1/cm]
    const double musp_up, // Reduced scattering coefficient in the upper layer [1/cm]
    const double msd_dn,  // Mean-square displacement in the lower layer [cm^2]
    const double mua_dn,  // Absorption coefficient in the lower layer [1/cm]
    const double musp_dn, // Reduced scattering coefficient in the lower layer [1/cm]
    const double n,       // Ratio of the refractive index of the medium to that of the surrounding medium
    const double d,       // Thickness of the upper layer [cm]
    const double lambda0, // Wavelength of light in vacuum [nm]
    const double q        // Spatial frequency [1/cm]
    ) {
    const double lambda0_cm = lambda0 * 1e-7; // Convert to cm
    const double k0 = 2 * M_PI / lambda0_cm;
    const double z0 = 1.0 / musp_up;
    const double r = -1.44 / (n * n) + 0.71 / n + 0.668 + 0.0636 * n;
    const double a = (1 + r) / (1 - r);
    const double zb = 2 * a / (3 * musp_up);

    const double xi_up = sqrt(q * q + 3 * mua_up * musp_up + k0 * k0 * musp_up * musp_up * msd_up);
    const double xi_dn = sqrt(q * q + 3 * mua_dn * musp_dn + k0 * k0 * musp_dn * musp_dn * msd_dn);

    const double term1 = 3 * musp_up * sinh(xi_up * (z0 + zb)) / xi_up;
    const double num = xi_up * cosh(xi_up * d) / (3 * musp_up) + xi_dn * sinh(xi_up * d) / (3 * musp_dn);
    const double den = xi_up * cosh(xi_up * (d + zb)) / (3 * musp_up) + xi_dn * sinh(xi_up * (d + zb)) / (3 * musp_dn);
    const double term2 = 3 * musp_up * sinh(xi_up * z0) / xi_up;

    return term1 * (num / den) - term2;
}


/*
 * Integrand for the G1 calculation in a bilayer medium. To be passed to scipy.quad as a low-level C function.
*/
double integrand(int n, const double *x, const void *params) {
    const double q = x[0];
    const double *p = params;
    const double msd_up = p[0];
    const double mua_up = p[1];
    const double musp_up = p[2];
    const double msd_dn = p[3];
    const double mua_dn = p[4];
    const double musp_dn = p[5];
    const double n_refractive = p[6];
    const double d = p[7];
    const double lambda0 = p[8];
    const double rho = p[9];

    const double g1_sf = g1_spatial_freq(
        msd_up,
        mua_up,
        musp_up,
        msd_dn,
        mua_dn,
        musp_dn,
        n_refractive,
        d,
        lambda0,
        q
    );

    return g1_sf * q * gsl_sf_bessel_J0(q * rho);
}