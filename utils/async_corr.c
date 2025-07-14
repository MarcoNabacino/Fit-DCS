#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int64_t unique_with_inverse(
    const int64_t* sorted_arr,
    int64_t* unique_vals,
    int64_t* inverse_indices,
    const size_t n
) {
    if (n == 0) return 0;

    int64_t unique_count = 0;
    unique_vals[0] = sorted_arr[0];
    inverse_indices[0] = 0;

    for (size_t i = 1; i < n; i++) {
        if (sorted_arr[i] != sorted_arr[i - 1]) {
            unique_count++;
            unique_vals[unique_count] = sorted_arr[i];
        }
        inverse_indices[i] = unique_count;
    }

    return unique_count + 1;
}


static void bincount_weighted(
    const int64_t* inverse_indices,
    const int64_t* weights_in,
    const int n_input,
    int64_t* weights_out,
    const int n_unique
) {
    memset(weights_out, 0, n_unique * sizeof(int64_t));
    for (int i = 0; i < n_input; i++) {
        const int64_t idx = inverse_indices[i];
        weights_out[idx] += weights_in[i];
    }
}


void async_corr(
    const int64_t* t_raw,
    int n_tags,
    const int p, const int m, const int s,
    const double tau_start,
    const double t0,
    double* g2_out,
    double* tau_out
) {
    if (p % m != 0) return;

    const int n_overlapped_bins = p / m;
    const int n_bins = (p - n_overlapped_bins) * s;

    int64_t* t = malloc(n_tags * sizeof(int64_t));
    int64_t* weights = malloc(n_tags * sizeof(int64_t));
    for (int i = 0; i < n_tags; i++) {
        t[i] = t_raw[i];
        weights[i] = 1;
    }
    const int64_t dt = t[n_tags - 1] - t[0];
    const int64_t shift_start = (int64_t)(tau_start / t0);

    const double mean_rate = (double)n_tags / (double)dt;

    int64_t* autocorr = calloc(n_bins, sizeof(int64_t));
    int64_t* autotime = calloc(n_bins, sizeof(int64_t));
    int64_t* bin_width = malloc(n_bins * sizeof(int64_t));

    int64_t delta = 1;
    int64_t shift = 0;
    int tau_index = 0;

    for (int stage = 0; stage < s; stage++) {
        // Handle duplicates
        int64_t* t_unique = malloc(n_tags * sizeof(int64_t));
        int64_t* inverse = malloc(n_tags * sizeof(int64_t));
        int64_t* weights_new = malloc(n_tags * sizeof(int64_t));

        const int n_unique = (int)unique_with_inverse(t, t_unique, inverse, n_tags);
        bincount_weighted(inverse, weights, n_tags, weights_new, n_unique);

        free(t);
        free(weights);
        t = t_unique;
        weights = weights_new;
        n_tags = n_unique;
        free(inverse);

        for (int b = n_overlapped_bins; b < p; b++) {
            shift += delta;
            const int64_t lag = shift / delta;

            autotime[tau_index] = shift;
            bin_width[tau_index] = delta;

            if (shift < shift_start) {
                tau_index++;
                continue;
            }

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

        delta *= m;
        for (int i = 0; i < n_tags; i++) {
            t[i] /= m;
        }
    }

    const double norm_factor = 1 / (mean_rate * mean_rate);
    for (int i = 0; i < tau_index; i++) {
        if (dt - autotime[i] > 0) {
            g2_out[i] = norm_factor * (double)autocorr[i] / (double)bin_width[i] / (double)(dt - autotime[i]);
        } else {
            g2_out[i] = 0.0;
        }
        tau_out[i] = (double)autotime[i] * t0;
    }

    free(t);
    free(weights);
    free(autocorr);
    free(autotime);
    free(bin_width);
}