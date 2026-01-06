#!/usr/bin/env python3

import sys
import scipy
import numpy as np
import pennylane as qml
from pyscf import gto, scf

np.random.seed(42)

class VQE:
    def __init__(self, N=8, R=1.0):
        self.N = N
        self.R = R

        self.L = 3
        self.M = int(np.log2(self.N))
        self.wires = list(range(self.M))
        self.bases = np.array([list(np.binary_repr(i, width=self.M)) for i in range(self.N)], dtype='int')

        self.projector = self.gen_projector()

        self.dev = qml.device('default.qubit', wires=self.M)

    def gen_projector(self):
        X  = np.array([[0,   1], [1,  0]], dtype='complex')
        Y  = np.array([[0, -1j], [1j, 0]], dtype='complex')
        iY = 1j * Y 

        projector = []
        for i, j in [(i, j) for i in self.bases for j in self.bases]:
            p_ij = np.eye(self.N, dtype='complex')

            for mu in wires:
                Il = np.eye(2 ** mu)
                Ir = np.eye(2 ** (M - mu - 1))

                sign = (i[mu] << 1) - 1
                if i[mu] ^ j[mu]: p = (X - sign * iY) / 2
                else:             p = (X - sign * iY) @ (X + sign * iY) / 4
                p_ij @= np.kron(np.kron(Il, p), Ir)
            p_ij = (p_ij + p_ij.T.conj()) / 2
            projector.append(qml.Hermitian(p_ij, wires=self.wires))

        return projector

    def prep_circuit(self, theta, basis):
        qml.BasisState(basis, wires=wires)
        for m in wires: qml.RY(theta[m], wires=m)
        for n in range(L):
            for m in wires[:-1]: qml.CNOT(wires=[m, m+1])
            for m in wires:      qml.RY(theta[M * (n+1) + m], wires=m)

    @qml.qnode(self.dev, interface='autograd')
    def circuit_gamma(theta, basis):
        prep_circuit(theta, basis)
        return [qml.expval(p) for p in self.projector] #qml.state()

    @qml.qnode(self.dev, interface='autograd')
    def circuit_energy(theta, basis, hamiltonian):
        prep_circuit(theta, basis)
        return qml.expval(hamiltonian)

    def cost(theta):
        gamma = np.array([[x for x in circuit_gamma(theta, basis)] for basis in self.bases[:N]]).reshape(-1, N, N)
        dm = np.einsum('ij,kjl,ln->kin', s1e, gamma, s1e.T)[:N//2].sum(axis=0)
        vhf = mf.get_veff(cell, dm)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        return e_tot

vqe = VQE(N=4, R=1.0)

cell = gto.M(atom = '; '.join([f'H {R*i:.1f} 0 0' for i in range(N)]), # Gaussian-Type Orbital (GTO)
             basis = 'sto-3g',
             pseudo = 'gth-lda')

mf = scf.RKS(cell, xc='lda') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
#mf.kernel(); sys.exit(1)

s1e = mf.get_ovlp(cell)
h1e = mf.get_hcore(cell)

theta = np.random.rand(M * (L+1))
e_tot = cost(theta, bases[:N], projector)

res = scipy.optimize.minimize(cost, theta,
                              args=(bases[:N], projector),
                              method='L-BFGS-B',
                              options={'disp': True})

theta = res.x
print(theta)

"""
dm = mf.get_init_guess(cell, key='minao')

for i in range(max_iter):
    dm_last = dm
    vhf_last = vhf
    e_tot_last = e_tot

    fock = mf.get_fock(h1e, s1e, vhf, dm) # fock = h1e + vhf
    hamiltonian = qml.Hamiltonian(np.ravel(fock), projector)

    gamma = np.array([[x.item() for x in circuit_gamma(theta, basis, projector)] for basis in bases[:N]]).reshape(-1, N, N)
    dm = np.einsum('ij,kjl,ln->kin', s1e, gamma, s1e.T).sum(axis=0)
    vhf = mf.get_veff(cell, dm, dm_last, vhf_last)

    theta, e_tot = opt.step_and_cost(cost, theta, bases=bases[:N], projector=projector)

    #print(f'iter= {i+1:3d} E= {e_tot:10.8f} delta_E= {e_tot - e_tot_last:10.8f} |ddm|= {norm_ddm:10.8f}')
    print(f'iter= {i+1:3d} E= {e_tot:10.8f}')
"""


"""
for itr in range(50):
    dm_last = dm
    e_tot_last = e_tot

    fock = mf.get_fock(h1e, s1e, vhf, dm) # fock = h1e + vhf
    energy, coeff = mf.eig(fock, s1e)
    occ = mf.get_occ(energy, coeff)
    dm = mf.make_rdm1(coeff, occ)
    vhf = mf.get_veff(cell, dm, dm_last, vhf)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    norm_ddm = np.linalg.norm(dm - dm_last)
    print(f'iter= {itr+1:3d} E= {e_tot:9f} delta_E= {e_tot - e_tot_last:9f} |ddm|= {norm_ddm:9f}')

    if abs(e_tot - e_tot_last) < 1e-6: break

energy, coeff = mf.eig(fock, s1e)
occ = mf.get_occ(energy, coeff)
dm, dm_last = mf.make_rdm1(coeff, occ), dm
vhf = mf.get_veff(cell, dm, dm_last, vhf)
e_tot, e_tot_last = mf.energy_tot(dm, h1e, vhf), e_tot

fock = mf.get_fock(h1e, s1e, vhf, dm)
norm_ddm = np.linalg.norm(dm - dm_last)
print(f'iter= {"-":>3s} E= {e_tot:9f} delta_E= {e_tot - e_tot_last:9f} |ddm|= {norm_ddm:9f}')
"""

