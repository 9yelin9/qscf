#!/usr/bin/env python3

import numpy as np
#import pyscf.pbc.gto as gto
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import pyscf_ase
from ase.build import bulk
import matplotlib.pyplot as plt

si = bulk('Si', 'diamond', a=5.459)

cell = gto.Cell()
#cell.atom = pyscf_ase.ase_atoms_to_pyscf(si)
#cell.a = np.array(si.cell)
a = 5.459
cell.atom = f'Si 0 0 0; Si {a/4} {a/4} {a/4}'
cell.a = [[0, a/2, a/2],[a/2, 0, a/2],[a/2, a/2, 0]]
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 3
cell.build()

bp = si.cell.bandpath('LGXUKG', npoints=100)
#
# band structure from Gamma point sampling
#
mf = scf.RKS(cell).run()

band_kpts = cell.get_abs_kpts(bp.kpts)
e_kn = mf.get_bands(band_kpts)[0]

#
# band structure from 222 k-point sampling
#
#kmf = cell.KRKS(kpts=cell.make_kpts([2,2,2])).run()
#e_kn_2 = kmf.get_bands(band_kpts)[0]

nocc = cell.nelectron // 2
au2ev = 27.21139
#e_kn = (np.array(e_kn) - mf.get_fermi()) * au2ev
#e_kn_2 = (np.array(e_kn_2) - kmf.get_fermi()) * au2ev
e_kn = (np.array(e_kn)) * au2ev
#e_kn_2 = (np.array(e_kn_2)) * au2ev

fig, ax = plt.subplots(figsize=(5, 6))
pyscf_ase.plot_band_structure(bp, e_kn, ax, 'black')
#pyscf_ase.plot_band_structure(bp, e_kn_2, ax, 'blue')

emin = -1*au2ev
emax = 1*au2ev
ax.set_ylim(emin, emax)

plt.tight_layout()
plt.show()
