# Numerical Quantum Mechanics

#### **Lanczos Algorithm Implementation for 1D Quantum Mechanics**

This repository contains the source code, data, and final report for the **Numerical Quantum Mechanics** project. The project develops and validates a numerical framework for solving quantum mechanical eigenvalue problems using a matrix formulation and the Lanczos algorithm. Both serial and parallel implementations are provided, with the parallel versions leveraging MPI and a hybrid MPI+OpenMP approach to achieve strong scaling performance on modern high-performance computing systems.

## Overview

This project presents a comprehensive framework for tackling quantum mechanical eigenvalue problems via:

- **Matrix Formulation:** Discretization of the Schrödinger equation transforms the continuous problem into a large, sparse matrix eigenvalue problem.
- **Lanczos Algorithm:** An efficient iterative method is implemented to compute the lowest eigenvalues and corresponding eigenvectors.
- **Serial and Parallel Implementations:**
  - The **serial version** employs a finite-difference scheme with full reorthogonalization using the Modified Gram–Schmidt process.
  - The **parallel version** includes an MPI-only implementation (tested on Google Colab) and a hybrid MPI+OpenMP implementation (optimized and benchmarked on the Stromboli Cluster).

## Key Features

- **High-Performance Computing (HPC):** The hybrid MPI+OpenMP implementation efficiently exploits modern HPC architectures.
- **Efficiency and Accuracy:** Detailed convergence studies show rapid convergence of the Lanczos algorithm with minimal relative error.
- **Robust Domain Decomposition:** Effective management of ghost cell exchanges ensures data consistency during finite-difference computations.
- **Scalability Analysis:** Strong scaling studies are provided for various grid sizes and Krylov subspace dimensions, with performance benchmarks for CPU, GPU, and TPU runtimes on Google Colab and for the hybrid implementation on the Stromboli Cluster.

## Getting Started

### Prerequisites

- **Python 3** (with packages such as `numpy`, `mpi4py`, `scipy`, and `matplotlib`)
- **C Compiler with OpenMP Support:** e.g., `gcc`
- **LaTeX Distribution:** [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/) (ensure required packages like `minted` or `listings` are installed)
- For compiling the **C/OpenMP kernels**, run:

    ```bash
    gcc -fPIC -shared -O3 -fopenmp -o lanczos_openmp.so lanczos_openmp.c
    ```


### Installation and Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/dhruvpatel09/numerical-quantum-mechanics.git
    ```

2. **Navigate to the Repository:**

    ```bash
    cd numerical-quantum-mechanics
    ```

3. **Compile the C/OpenMP Kernels:**

    ```bash
    cd src/c-openmp
    gcc -fPIC -shared -O3 -fopenmp -o lanczos_openmp.so lanczos_openmp.c
    cd ../../
    ```

4. **(Optional) Install Python Dependencies:**
Ensure that you have installed `numpy`, `mpi4py`, `scipy`, and `matplotlib`:

    ```bash
    pip install numpy mpi4py scipy matplotlib
    ```

## Usage

### Running the Serial Code

For the serial implementation, open the Colab notebook in `src/colab-notebooks/serial/Matrix_QM_serial.ipynb` or run:

```python
python src/colab-notebooks/serial/matrix_qm_serial.py
```

### Running the Parallel Code

#### On Google Colab (MPI-only Implementation)

- Navigate to `src/colab-notebooks/parallel/Matrix_QM_parallel.ipynb` for an interactive notebook.

#### On the Stromboli Cluster (Hybrid MPI+OpenMP Implementation)

- Use the provided SLURM submission script located in the `scripts/` directory:

    ```bash
    sbatch scripts/submit_strong.sh
    ```

    This script submits jobs with varying MPI process counts for strong scaling studies.

- Alternatively, execute the parallel script interactively:

    ```bash
    mpirun -np <num_processes> python src/python-mpi/matrix_qm_parallel_mpi_omp.py --N ${<grid_size>} --m ${<krylov_size>} --output_dir ${<output_directory>}
    ```

    or

    ```bash
    srun python src/python-mpi/matrix_qm_parallel_mpi_omp.py --N ${<grid_size>} --m ${<krylov_size>} --output_dir ${<output_directory>}  
    ```

    For testing, use smaller values of `N`, `m`, and a designated output directory.

## Report

The full project report is available in the `report/` directory as both a LaTeX source (`matrix_qm_report.tex`) and a compiled PDF (`matrix_qm_report.pdf`). The report details the theoretical foundations, numerical methods, parallel implementations, scaling studies, and conclusions of the project.

## Data

The `data/` directory contains the source code and output files from various experiments:

- **Colab Serial and Parallel:** Eigenvalues, relative error data, and related plots.
- **Stromboli Parallel:** Logs, scaling results, and various configuration outputs for different grid sizes and Krylov subspace dimensions.

## References

The project builds on established research in quantum mechanics and numerical methods. For a complete list of references, see the Bibliography section in the report.

## Authors and Acknowledgment

Maintainers / Developers:

- **Dhruv Patel:** dhruv.patel[at]uni-wuppertal.de

- **Mohammadreza Khansari:** mohammadreza.khansari[at]uni-wuppertal.de

Advisor:

- **Dr. Tomasz Korzec:** korzec[at]uni-wuppertal.de

## Contributions

Contributions to improve this project are welcome. Feel free to fork the repository and submit a pull request. For questions or suggestions, please open an issue or contact me via [GitHub](https://github.com/dhruvpatel09/).

## License

This project is licensed under the terms of the [MIT License](https://choosealicense.com/licenses/mit/).

##
