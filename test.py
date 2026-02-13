#!/usr/bin/env python3

import numpy as np
from pyscf.pbc import gto, scf

mol = gto.M(atom = 'Si 0 0 0',
            a = np.eye(3) * 3.9,
            basis = 'gth-dzv',
            pseudo = 'gth-lda')
mf = scf.RKS(mol, xc='lda') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
mf.kernel()
