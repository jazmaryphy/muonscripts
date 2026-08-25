#!/bin/bash -l
#SBATCH --job-name=UCG_n8
#SBATCH --time=150:58:00
#SBATCH -n 96
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --err=8.e
#SBATCH --out=8.o

module load QuantumESPRESSO/7.3.1-foss-2023a
#module load QuantumESPRESSO/7.5-foss-2025a

export OMP_NUM_THREADS=2
export OMPI_MCA_osc=^ucx
export OMPI_MCA_btl=^openib,ofi
export OMPI_MCA_pml=^ucx
export OMPI_MCA_mtl=^ofi

mpirun pw.x -npool 2 -inp 8.in >> 8.out
