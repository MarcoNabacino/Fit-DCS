/*
 * Fit-DCS: A Python toolbox for Diffuse Correlation Spectroscopy analysis
 * Copyright (C) 2025  Marco Nabacino
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <stdint.h>
#include <stdlib.h>
#include <string.h>


/*
 * This function takes a sorted array of time tags and returns the unique values, along with their inverse indices.
 * For example, if the input is [1, 1, 2, 3, 3], the output will be:
 * unique_vals = [1, 2, 3], inverse_indices = [0, 0, 1, 2, 2].
 * The function returns the number of unique values found.
*/
static int64_t unique_with_inverse(
    const int64_t* sorted_arr,  // Sorted (in ascending order) array of time tags
    int64_t* unique_vals,       // Output array for unique values
    int64_t* inverse_indices,   // Output array for inverse indices
    const size_t n              // Number of elements in the sorted array
) {
    if (n == 0) return 0;

    unique_vals[0] = sorted_arr[0]; // The first element is always unique
    inverse_indices[0] = 0;

    int64_t unique_count = 1;

    for (size_t i = 1; i < n; i++) {
        if (sorted_arr[i] != sorted_arr[i - 1]) {
            // Found a new unique value
            unique_vals[unique_count++] = sorted_arr[i];
        }
        inverse_indices[i] = unique_count - 1;
    }

    return unique_count;
}


/*
 * This function counts the occurrences of each unique value in the input array and sums their weights.
 * It uses the inverse indices to map each input value to its corresponding unique index.
 * The output is an array where each index corresponds to a unique value and contains the sum of weights for that value.
 */
static void bincount_weighted(
    const int64_t* inverse_indices, // Array of inverse indices mapping input values to unique indices
    const int64_t* weights_in,      // Array of weights corresponding to the input values
    const int n_input,              // Number of elements in the input arrays
    int64_t* weights_out,           // Output array for summed weights, must be preallocated with size n_unique
    const int n_unique              // Number of unique values, size of weights_out
) {
    memset(weights_out, 0, n_unique * sizeof(int64_t)); // Initialize output weights to zero
    for (int i = 0; i < n_input; i++) {
        // For each input value, find its unique index and add the corresponding weight.
        const int64_t idx = inverse_indices[i];
        weights_out[idx] += weights_in[i];
    }
}


/*
 * Calculates the autocorrelation function using the asynchronous algorithm described in [1].
 *
 * [1] Wahl, M. et al. (2003), "Fast calculation of fluorescence correlation data with asynchronous time-correlated
 * single photon counting".
 */
void async_corr(
    const int64_t* t_raw,   // Array of time tags (unit specified by t0)
    int n_tags,             // Number of time tags
    const int p,            // Number of bins in each linear correlator
    const int m,            // Binning ratio. Must divide p.
    const int s,            // Number of stages (linear correlators)
    const double tau_start, // Start time for the correlation (unit specified by t0)
    const double t0,        // Time unit, in seconds. For Swabian time taggers this should be 1e-12, i.e., 1 ps.
    double* g2_out,         // Output array for the g2 correlation values
    double* tau_out         // Output array for the time lags corresponding to g2_out
) {
    if (p % m != 0) return;     // Ensure that p is divisible by m

    const int n_overlapped_bins = p / m; // Number of bins of each stage that overlap with the next stage
    const int n_bins = (p - n_overlapped_bins) * s; // Total number of bins in the output

    int64_t* t = malloc(n_tags * sizeof(int64_t));
    int64_t* weights = malloc(n_tags * sizeof(int64_t));
    for (int i = 0; i < n_tags; i++) {
        t[i] = t_raw[i];
        weights[i] = 1;
    }
    const int64_t dt = t[n_tags - 1] - t[0]; // Measurement duration
    const int64_t shift_start = (int64_t)(tau_start / t0); // Start time for the correlation in time tagger units

    const double mean_rate = (double)n_tags / (double)dt; // Mean countrate of the measurement, needed for normalization

    // Initialize output arrays
    int64_t* autocorr = calloc(n_bins, sizeof(int64_t)); // Unnormalized autocorrelation
    int64_t* autotime = malloc(n_bins * sizeof(int64_t)); // Autocorrelation time lags in time tagger units
    int64_t* bin_width = malloc(n_bins * sizeof(int64_t)); // Width of each bin in time tagger units

    int64_t delta = 1; // Bin width in time tagger units
    int64_t shift = 0; // Lag in time tagger units
    int tau_index = 0;

    for (int stage = 0; stage < s; stage++) {
        // Handle duplicates, by removing them and summing their weights
        int64_t* t_unique = malloc(n_tags * sizeof(int64_t)); // Array for unique time tags
        int64_t* inverse = malloc(n_tags * sizeof(int64_t)); // Array for inverse indices
        int64_t* weights_new = malloc(n_tags * sizeof(int64_t)); // Array for updated weights

        const int n_unique = (int)unique_with_inverse(t, t_unique, inverse, n_tags);
        bincount_weighted(inverse, weights, n_tags, weights_new, n_unique);
        free(inverse);

        n_tags = n_unique;
        // Update the time tags to the unique values and their corresponding weights
        free(t);
        t = t_unique;
        free(weights);
        weights = weights_new;

        for (int b = n_overlapped_bins; b < p; b++) {
            shift += delta; // Increment the shift for the current bin
            const int64_t lag = shift / delta; // Lag in number of bins

            autotime[tau_index] = shift; // Store the current shift in autotime
            bin_width[tau_index] = delta; // Store the width of the current bin

            if (shift < shift_start) {
                // If the shift is less than the start time, skip this bin
                tau_index++;
                continue;
            }

            // Calculate the autocorrelation for the current bin
            int i = 0, j = 0;
            while (i < n_tags && j < n_tags) {
                const int64_t a1 = t[i];
                const int64_t a2 = t[j] + lag;
                if (a1 < a2) {
                    i++;
                } else if (a2 < a1) {
                    j++;
                } else {
                    autocorr[tau_index] += weights[i] * weights[j];
                    i++;
                    j++;
                }
            }

            tau_index++;
        }

        delta *= m; // Increase the bin width for the next stage
        // Coarsen the time tags by a factor of m for the next stage
        for (int i = 0; i < n_tags; i++) {
            t[i] /= m;
        }
    }

    // Normalize the autocorrelation values and prepare the output
    const double norm_factor = 1 / (mean_rate * mean_rate);
    for (int i = 0; i < tau_index; i++) {
        if (dt - autotime[i] > 0) {
            g2_out[i] = norm_factor * (double)autocorr[i] / (double)bin_width[i] / (double)(dt - autotime[i]);
        } else {
            g2_out[i] = 0.0;
        }
        tau_out[i] = (double)autotime[i] * t0;
    }

    // Free allocated memory
    free(t);
    free(weights);
    free(autocorr);
    free(autotime);
    free(bin_width);
}