#!/bin/bash
# submit_strong.sh
# Strong Scaling Study on Stromboli Cluster
# Author: Dhruv Patel (2130292), Mohammadreza Khansari (2132180)
# Date: 31st March 2025

#SBATCH --partition=compute2011

N=$((2**20 + 1))  # 1,048,577 grid points (odd for symmetry)
m=60		  # Krylov subspace size

OUTPUT_DIR="strong_scaling_results"
CORES_PER_NODE=24 # Stromboli cluster configuration

# Looping over different process counts for strong scaling study
for P in 2 4 8 16 32 64; do
    if [ $P -eq 64 ]; then
        # For 64 MPI tasks, use 3 nodes; do not force tasks-per-node to let Slurm optimize allocation
        NODES=3
        TASKS_PER_NODE=""
    else
        # Calculating the number of nodes required and tasks per node for other cases
        NODES=$(( (P + CORES_PER_NODE - 1) / CORES_PER_NODE ))
        TASKS_PER_NODE=$(( (P + NODES - 1) / NODES ))
    fi

    echo "Submitting P=$P (nodes=$NODES)..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=strong_P${P}
#SBATCH --output=strong_P${P}.log
#SBATCH --ntasks=${P}
#SBATCH --nodes=${NODES}
#SBATCH --partition=compute2011
#SBATCH --cpus-per-task=1

# OpenMP settings: using one thread per MPI process
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Disabling the internal multithreading for libraries to avoid oversubscription
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Stability and performance flags
export MKL_SERIAL=yes
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

# Additional environment settings to ensure consistency
export SLURM_EAR_LOAD_MPI_VERSION="intel"
export MKL_DOMAIN_ALL=1
export MKL_THREADING_LAYER=sequential

# Executing the parallel Lanczos algorithm
srun python ./matrix_qm_parallel_mpi_omp.py --N ${N} --m ${m} --output_dir ${OUTPUT_DIR}
EOF
done
