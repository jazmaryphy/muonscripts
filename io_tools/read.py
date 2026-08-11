# %%
import numpy as np
from io_tools.qe import read_efg
from io_tools.read_ase import read_from_file

# %%
def read_cif(filename):
    """# Load ('*.CIF' or '*.POSCAR' or '*.VASP') input files"""
    return read_from_file(filename=filename)  

def read_poscar(filename):
    """# Load ('*.CIF' or '*.POSCAR' or '*.VASP') input files"""
    return read_from_file(filename=filename)  

def read_vasp(filename):
    """# Load ('*.CIF' or '*.POSCAR' or '*.VASP') input files"""
    return read_from_file(filename=filename)  

def read_elk_geom(filename):
    """# Load ('GEOMETRY.OUT') output from Elk"""
    return read_from_file(filename=filename, format='elk')  

def read_qe_in(filename):
    """Load QE *.pwi input file"""
    return read_from_file(filename=filename, format='espresso-in')

def read_qe_out(filename):
    """Load QE *.pwo output file"""
    return read_from_file(filename=filename, format='espresso-out', index=-1)

def read_qe_efg(filename):
    """Load QE EFG tensors from output file"""
    return read_efg(filename)
# %%
def read_elk_efg(filename):
    """# Load ('EFG.OUT') EFGs from Elk and convert to SI"""

    EFGs = []
    with open(filename, 'r') as f:
        text = f.readlines()
        for i, l in enumerate(text):
            if 'EFG tensor :' in l:
                tmp = 9.71736166E21 * np.fromstring(''.join((text[i+1],text[i+2],text[i+3])),sep=' ').reshape([3,3])
                """# Set numerically-zero tensor elements to exactly zero.
                # EFGs are typically of order 10^20 - 10^22 V/m^2 for nuclei in solids
                # set a tolerance for the zeroth 
                """
                ## hardcoded assumptions that anything 9 order of magnitude less are zero
                ## tmp[np.abs(tmp)<1e15] = 0.0  
                # tol = 1e-8 * np.max(np.abs(tmp))
                # tmp[np.abs(tmp) < tol] = 0.0
                EFGs.append(tmp)

    return np.array(EFGs)