// lanczos_openmp.c
// Optimized C/OpenMP Kernels for Hamiltonian and Gram-Schmidt
// Compile: gcc -fPIC -shared -O3 -fopenmp -o lanczos_openmp.so lanczos_openmp.c
// Author: Dhruv Patel (2130292), Mohammadreza Khansari (2132180)
// Date: 31st March 2025

#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------------------
// Function: Hv
// Purpose: Apply the Hamiltonian operator to a vector segment using finite
//          differences. This includes both the kinetic (second derivative)
//          and potential (harmonic oscillator potential) energy terms
// -------------------------------------------------------------------------
void Hv(double* v, double a, double* x_local, int local_n, double* result) {
    const double hbar2 = 1.0;
    const double coeff = hbar2 / (2 * a * a);
    int j;

    // Kinetic Energy: Central Difference approximation
    #pragma omp parallel for default(none) shared(v, result, local_n) private(j)
    for (j = 0; j < local_n; j++) {
        result[j] = coeff * (v[j] + v[j + 2] - 2 * v[j + 1]);
    }

    // Potential Energy: Harmonic oscillator potential
    #pragma omp parallel for default(none) shared(v, x_local, result, local_n) private(j)
    for (j = 0; j < local_n; j++) {
        result[j] += 0.5 * x_local[j] * x_local[j] * v[j + 1];
    }
}

// -------------------------------------------------------------------------
// Function: modified_gram_schmidt
// Purpose: Apply the Modified Gram-Schmidt process to orthogonalize the
//          Lanczos basis vectors stored in a flattened matrix
// Parameters:
//    matrix: Pointer to the flattened matrix (column-major order)
//    n: Number of rows (grid points for the current process)
//    k: Number of Lanczos vectors (columns) to orthogonalize
// -------------------------------------------------------------------------
void modified_gram_schmidt(double* matrix, int n, int k) {
    int col, row, other_col;
    for (col = 0; col < k; col++) {
        double norm = 0.0;
        // Computing the norm of the current column vector
        #pragma omp parallel for reduction(+:norm) default(none) shared(matrix, n, col) private(row)
        for (row = 0; row < n; row++) {
            norm += matrix[row + col * n] * matrix[row + col * n];
        }
        norm = sqrt(norm);

        // If norm is too small, then reinitializing to the identity basis (avoiding division by zero)
        if (norm < 1e-15) {
            #pragma omp parallel for default(none) shared(matrix, n, col) private(row)
            for (row = 0; row < n; row++) {
                matrix[row + col * n] = (row == col % n) ? 1.0 : 0.0;
            }
            norm = 1.0;
        }

        // Normalizing the column vector
        #pragma omp parallel for default(none) shared(matrix, n, col, norm) private(row)
        for (row = 0; row < n; row++) {
            matrix[row + col * n] /= norm;
        }

        // Orthogonalizing subsequent columns against the current one
        for (other_col = col + 1; other_col < k; other_col++) {
            double dot = 0.0;
            #pragma omp parallel for reduction(+:dot) default(none) shared(matrix, n, col, other_col) private(row)
            for (row = 0; row < n; row++) {
                dot += matrix[row + other_col * n] * matrix[row + col * n];
            }
            #pragma omp parallel for default(none) shared(matrix, n, col, other_col, dot) private(row)
            for (row = 0; row < n; row++) {
                matrix[row + other_col * n] -= dot * matrix[row + col * n];
            }
        }
    }
}
