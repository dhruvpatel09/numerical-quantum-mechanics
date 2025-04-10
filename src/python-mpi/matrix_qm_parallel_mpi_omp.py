# matrix_qm_parallel_mpi_omp.py
# Parallel Lanczos Algorithm for 1D Quantum Mechanics
# Hybrid MPI + OpenMP implementation with domain decomposition
# Author: Dhruv Patel (2130292), Mohammadreza Khansari (2132180)
# Date: 31st March 2025

import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import eigsh
from ctypes import CDLL, c_int, c_double, POINTER

# Loading C/OpenMP library for accelerated kernels
lib = CDLL('./lanczos_openmp.so')

# =============================================================================
# Defining C function signatures
# =============================================================================

lib.Hv.argtypes = [
    POINTER(c_double),  # Input vector v with ghost cells (for boundary exchange)
    c_double,           # Grid spacing 'a'
    POINTER(c_double),  # Local position array
    c_int,              # Local vector size (number of grid points in current process)
    POINTER(c_double)   # Output vector for the Hamiltonian result
]
lib.Hv.restype = None

lib.modified_gram_schmidt.argtypes = [
    POINTER(c_double),  # Flattened matrix data for Lanczos vectors
    c_int,              # Number of rows (local grid points)
    c_int               # Number of columns (current iteration count 'k')
]
lib.modified_gram_schmidt.restype = None

# =============================================================================
# Lanczos Algorithm
# =============================================================================

def Lanczos_algorithm(m, n, a, rank, num_procs, comm):
    """
    Implements the parallel Lanczos algorithm using MPI domain decomposition

    Each MPI process works on a local segment of the full wavefunction
    Communication is required to exchange boundary values between processes

    Args:
        m (int): Krylov subspace size (iteration)
        n (int): Total number of grid points
        a (float): Grid spacing
        rank (int): MPI rank of the current process
        num_procs (int): Total number of MPI processes
        comm (MPI.Comm): MPI communicator

    Returns:
        eigenvalues (ndarray): 10 smallest eigenvalues (root process only)
    """

    # Domain decomposition: Splitting grid points across processes
    sizes = distribute_elements(n, num_procs)
    local_n = sizes[rank]
    starts = [sum(sizes[:i]) for i in range(num_procs)]
    start_idx = starts[rank]

    # Precomputing the local position array for current process
    M = (n - 1) // 2
    x_local = a * np.arange(start_idx - M, start_idx - M + local_n)

    # Initializing Lanczos vectors and coefficients
    v = np.zeros((local_n, m), dtype=np.float64)   # Orthonormal basis
    B = np.zeros(m - 1, dtype=np.float64)	   # Off-diagonal elements
    alpha = np.zeros(m, dtype=np.float64)	   # Diagonal elements

    # Initializing the starting vector (local part) and normalizing it globally
    v1 = np.random.randn(local_n)
    v1 = normalize_vector(v1, comm)

    # First Lanczos iteration: applying Ĥ using boundary exchange
    v_temp = exchange_boundary_data(v1, local_n, rank, num_procs, comm)
    w1_p = Hv(v_temp, a, x_local, comm)
    a1 = dot_product(w1_p, v1, comm)
    w1 = w1_p - a1 * v1
    alpha[0] = a1
    v[:, 0] = v1

    comm.Barrier()
    start_time = MPI.Wtime()

    # Lanczos iteration loop
    for j in range(1, m):
        # Computing the beta coefficient with global norm
        B[j - 1] = get_global_norm(w1, comm)

        # Handling the Lanczos breakdown
        if np.isclose(B[j-1], 0.0):
            # Restart if numerical breakdown occurs
            v[:, j] = normalize_vector(np.random.randn(local_n), comm)
        else:
            v[:, j] = w1 / B[j-1]

        # Function call: C-accelerated Modified Gram-Schmidt reorthogonalization
        v = modified_gram_schmidt(v, j + 1, comm)

        # Applying the Hamiltonian operator (with boundary communication)
        v_temp = exchange_boundary_data(v[:, j], local_n, rank, num_procs, comm)
        w1_p = Hv(v_temp, a, x_local, comm)
        alpha[j] = dot_product(w1_p, v[:, j], comm)
        w1 = w1_p - alpha[j]*v[:, j] - B[j - 1]*v[:, j - 1]

    comm.Barrier()
    iteration_time = MPI.Wtime() - start_time

    # Root process (rank 0) solves tridiagonal system and evaluate eigenvalues
    if rank == 0:
        T = np.diag(alpha) + np.diag(B, 1) + np.diag(B, -1)
        eigenvalues, _ = eigsh(T, k=10, which='SA')
    else:
        eigenvalues = None

    return eigenvalues

# =============================================================================
# MPI Helper Functions
# =============================================================================

def distribute_elements(N, num_procs):
    """
    Distributes grid points evenly across MPI processes

    Returns:
        sizes (list): Number of points per process
    """
    base, rem = divmod(N, num_procs)
    return [base + 1 if i < rem else base for i in range(num_procs)]

def exchange_boundary_data(v, local_n, rank, size, comm):
    """
    Performs non-blocking communication to exchange ghost cell data with neighboring processes

    This is needed because the Hamiltonian operator (finite differences) accesses neighboring elements

    Args:
        v (ndarray): Local vector segment
        local_n (int): Local dimension
        rank (int): Current process rank
        size (int): Total number of processes
        comm (MPI.Comm): MPI communicator

    Returns:
        ndarray: Vector with ghost cells from neighbors
    """
    right_rank = (rank + 1) % size
    left_rank = (rank - 1) % size

    # Preparing the send/receive buffers
    send_buf = np.array([v[0], v[-1]], dtype=np.float64)
    recv_buf = np.empty(2, dtype=np.float64)

    # Non-blocking communications
    comm_time = -MPI.Wtime()
    reqs = [
        comm.Isend(send_buf[0:1], dest=left_rank),
        comm.Isend(send_buf[1:2], dest=right_rank),
        comm.Irecv(recv_buf[0:1], source=right_rank),
        comm.Irecv(recv_buf[1:2], source=left_rank)
    ]
    MPI.Request.Waitall(reqs)
    comm_time += MPI.Wtime()

    # Building the extended local vector with ghost cells at boundaries
    v_ext = np.empty(local_n + 2, dtype=np.float64)
    v_ext[1:-1] = v		# Local data
    v_ext[0] = recv_buf[0]	# Left ghost cell
    v_ext[-1] = recv_buf[1]	# Right ghost cell

    if rank == 0:
        print(f"Comm time per step: {comm_time:.6f}s")
    return v_ext

def normalize_vector(v, comm):
    """Normalizes the vector using a global norm computed via MPI"""
    return v / get_global_norm(v, comm)

def get_global_norm(v, comm):
    """Computes the L2 norm of a vector across all MPI processes"""
    local_sq = np.sum(v**2)
    global_sq = comm.allreduce(local_sq, op=MPI.SUM)
    return np.sqrt(global_sq)

def dot_product(v, u, comm):
    """Computes the global dot product of two vectors across MPI processes"""
    local_dot = v @ u
    return comm.allreduce(local_dot, op=MPI.SUM)

# =============================================================================
# Python wrappers for calling C routines
# =============================================================================

def Hv(v, a, x_local, comm):
    """
    Applies the Hamiltonian operator using the optimized C/OpenMP kernel
    Wraps the C function call for applying both kinetic and potential parts

    Args:
        v (ndarray): Local segment of ψ with ghost cells
        a (float): Grid spacing
        x_local (ndarray): Local position array
        comm (MPI.Comm): MPI communicator

    Returns:
        result (ndarray): Local segment of Hψ
    """
    local_n = len(x_local)
    v_ptr = v.ctypes.data_as(POINTER(c_double))
    x_ptr = x_local.ctypes.data_as(POINTER(c_double))
    result = np.zeros(local_n, dtype=np.float64)
    result_ptr = result.ctypes.data_as(POINTER(c_double))

    # Calling C function via ctypes interface
    lib.Hv(v_ptr, c_double(a), x_ptr, c_int(local_n), result_ptr)
    return result

def modified_gram_schmidt(matrix, k, comm):
    """
    Applies the C-accelerated Modified Gram-Schmidt orthogonalization
    Reorthogonalizes the current Lanczos vectors

    Args:
        matrix (ndarray): Column vectors to orthogonalize
        k (int): Number of vectors to process
        comm (MPI.Comm): MPI communicator

    Returns:
        ndarray: Orthonormalized vectors
    """
    local_n, m = matrix.shape
    matrix_flat = np.ascontiguousarray(matrix[:, :k].flatten())
    matrix_ptr = matrix_flat.ctypes.data_as(POINTER(c_double))

    # Calling C function via ctypes interface
    lib.modified_gram_schmidt(matrix_ptr, c_int(local_n), c_int(k))

    # Reshaping back to original dimensions
    matrix[:, :k] = matrix_flat.reshape((local_n, k))
    return matrix

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid MPI+OpenMP Lanczos Algorithm')
    parser.add_argument('--N', type=int, required=True, help='Total grid points (e.g., 1048575)')
    parser.add_argument('--m', type=int, required=True, help='Krylov subspace size (e.g., 33)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Physical parameters
    L = 25.0		# System size [-L/2, L/2]
    N = args.N		# Number of spatial grid points
    m = args.m		# Krylov subspace size
    a = L / (N - 1)	# Grid spacing

    if rank == 0:
        import os
        os.makedirs(args.output_dir, exist_ok=True)

    comm.Barrier()
    start_time = MPI.Wtime()

    # Executing the algorithm to evaluate eigenvalues
    eigenvalues = Lanczos_algorithm(m, N, a, rank, size, comm)

    # Saving the results (root process only)
    if rank == 0:
        exec_time = MPI.Wtime() - start_time
        print(f"Execution time: {exec_time:.4f} seconds")
        np.savetxt(f'{args.output_dir}/N_{N}_P_{size}_m_{m}.txt', eigenvalues)
