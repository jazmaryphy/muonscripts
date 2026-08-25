#!/bin/bash -l
#SBATCH --job-name=UCG_n10
#SBATCH --time=150:58:00
#SBATCH -n 96
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --err=10.e
#SBATCH --out=10.o

module load QuantumESPRESSO/7.3.1-foss-2023a
#module load QuantumESPRESSO/7.5-foss-2025a

export OMP_NUM_THREADS=2
export OMPI_MCA_osc=^ucx
export OMPI_MCA_btl=^openib,ofi
export OMPI_MCA_pml=^ucx
export OMPI_MCA_mtl=^ofi

mpirun pw.x -npool 2 -inp 10.in >> 10.out
