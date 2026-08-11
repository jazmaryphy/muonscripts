#!/bin/bash -l
#SBATCH --job-name=UCG_p1
#SBATCH --time=150:58:00
#SBATCH -n 96
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --err=1.e
#SBATCH --out=1.o

module load QuantumESPRESSO/7.3.1-foss-2023a

export OMP_NUM_THREADS=2
export OMPI_MCA_osc=^ucx
export OMPI_MCA_btl=^openib,ofi
export OMPI_MCA_pml=^ucx
export OMPI_MCA_mtl=^ofi

mpirun pw.x -npool 2 -inp 1.in  >> 1.out
