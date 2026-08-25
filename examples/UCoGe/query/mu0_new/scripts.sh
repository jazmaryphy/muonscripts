for i in {1..10}; do
cat << EOF > ${i}.sh
#!/bin/bash -l
#SBATCH --job-name=UCG_n${i}
#SBATCH --time=150:58:00
#SBATCH -n 96
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --err=${i}.e
#SBATCH --out=${i}.o

module load QuantumESPRESSO/7.3.1-foss-2023a
#module load QuantumESPRESSO/7.5-foss-2025a

export OMP_NUM_THREADS=2
export OMPI_MCA_osc=^ucx
export OMPI_MCA_btl=^openib,ofi
export OMPI_MCA_pml=^ucx
export OMPI_MCA_mtl=^ofi

mpirun pw.x -npool 2 -inp ${i}.in >> ${i}.out
EOF
chmod +x ${i}.sh
done

# for i in {1..10}; do sbatch ${i}.sh; done
