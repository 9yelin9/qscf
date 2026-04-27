#!/usr/bin/env python3

import os
import jax
import sys
import time
import scipy
import pickle
import argparse
import numpy as np
import pennylane as qml
from jax import numpy as jnp
#from pennylane import numpy as pnp

np.random.seed(42)
np.set_printoptions(precision=3, linewidth=120, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('mat', type=str)
args = parser.parse_args()

class VQE:
    def __init__(self, mat, mf, output_tag, L=3):
        #self.mf.kernel(); sys.exit(1)
        self.mat = mat
        self.mf = mf
        self.output_tag = output_tag
        self.L = L

        self.N = self.mat.nao
        self.Ne = self.mat.nelectron
        n_full, n_half = divmod(self.Ne, 2)
        self.Nocc_max = n_full + n_half
        self.Nocc = [2]*n_full + [1]*n_half + [0]*(self.N - self.Nocc_max)
        print(f'N={self.N} Ne={self.Ne} Nocc={self.Nocc}')

        self.nkpts = len(self.mf.kpts)
        self.s1e = self.mf.get_ovlp(self.mat).reshape(self.nkpts, self.N, self.N)
        self.h1e = self.mf.get_hcore(self.mat).reshape(self.nkpts, self.N, self.N)
        print(f'nkpts={self.nkpts}')

        self.output_dir = f'output/{output_tag}'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'output_dir={self.output_dir}')

        self.M = int(np.ceil(np.log2(self.N)))
        self.wires = list(range(self.M))
        self.bases = np.array([list(np.binary_repr(i, width=self.M)) for i in range(self.N)], dtype='int')
        self.weight = [(self.N - k) / (self.N * (self.N + 1) / 2) for k in range(self.N)]
        print(f'L={self.L} M={self.M}', end='\n\n')

        #self.dev = qml.device('default.qubit', wires=self.M)
        self.dev = qml.device('lightning.gpu', wires=self.M)
        self.qnode_dm = qml.QNode(self.circuit_dm, self.dev)
        self.qnode_eks = qml.QNode(self.circuit_eks, self.dev)

    def gen_projector(self):
        p = {(0, 0): np.array([[1, 0], [0, 0]], dtype='complex'),
             (0, 1): np.array([[0, 1], [0, 0]], dtype='complex'),
             (1, 0): np.array([[0, 0], [1, 0]], dtype='complex'),
             (1, 1): np.array([[0, 0], [0, 1]], dtype='complex')}

        projector = []
        for i, j in [(i, j) for i in self.bases for j in self.bases]:
            p_ij = np.eye(1, dtype='complex')
            for m in self.wires: p_ij = np.kron(p_ij, p[(i[m], j[m])])
            p_ij = (p_ij + p_ij.T.conj())/2
            projector.append(qml.pauli_decompose(p_ij))
        return projector

    def prep_circuit(self, theta, basis):
        qml.BasisState(basis, wires=self.wires)
        for m in self.wires: qml.RY(theta[m], wires=m)
        for n in range(self.L):
            for m in self.wires[:-1]: qml.CNOT(wires=[m, m+1])
            for m in self.wires:      qml.RY(theta[self.M * (n+1) + m], wires=m)

    def circuit_dm(self, theta, basis, projector):
        self.prep_circuit(theta, basis)
        #return qml.density_matrix(self.wires)
        return [qml.expval(p) for p in projector]

    def circuit_eks(self, theta, basis, hamiltonian):
        self.prep_circuit(theta, basis)
        return qml.expval(hamiltonian)

    def get_dm(self, theta, projector, shots):
        theta = theta.reshape(self.nkpts, self.M * (self.L + 1))
        dm = np.zeros((self.nkpts, self.N, self.N), dtype=complex)
        for occ, basis in zip(self.Nocc, self.bases[:self.Nocc_max]):
            for k in range(self.nkpts):
                dm0 = np.array(self.qnode_dm(theta[k], basis, projector, shots=shots), dtype=float).reshape(self.N, self.N)
                #shp = scipy.linalg.fractional_matrix_power(self.s1e,  0.5)
                shm = scipy.linalg.fractional_matrix_power(self.s1e[k], -0.5)
                dm[k] += occ * (shm.conj().T @ ((dm0 + dm0.conj().T)/2) @ shm)
        #ne, occ = np.trace(self.s1e @ dm).real, np.linalg.eigvalsh(self.shp @ dm @ self.shp)
        #print(f'ne= {ne:9f} occ= {occ} => {sum(occ)}')
        return dm

    def cost_qdft(self, theta, projector, shots, fock, hamiltonian, alpha):
        eks = []
        for basis, weight in zip(self.bases[:self.Nocc_max], self.weight[:self.Nocc_max]):
            eks.append(self.qnode_eks(theta, basis, hamiltonian, shots=shots) * weight)
        eks = np.sum(eks)

        dm = self.get_dm(theta, projector, shots)
        R = fock @ dm @ self.s1e - self.s1e @ dm @ fock # FDS - SDF = 0
        return eks + alpha * np.linalg.norm(R, 'fro')**2

    def cost_qudft(self, theta, projector, shots):
        dm = self.get_dm(theta, projector, shots)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)
        return etot

    def cost_dm(self, theta, projector, shots, dm_target):
        dm = self.get_dm(theta, projector, shots)
        return np.linalg.norm(dm - dm_target)**2

    def init_dm(self, theta, projector, shots, max_iter=100, tol=1e-1):
        dm_target = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)

        t0 = time.time()
        res = scipy.optimize.minimize(self.cost_dm, theta,
                                      args=(projector, shots, dm_target),
                                      method='COBYLA',
                                      tol=tol,
                                      options={'disp': True})

        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.init_dm.__name__} time elapsed: {tm}m {ts}s', end='\n\n')
        return jnp.array(res.x, dtype=float, requires_grad=True), dm_target
        """
        opt = qml.RotosolveOptimizer()
        norm2 = 999

        print(f'\n{self.init_dm.__name__}\n{"iter":6s}{"norm2":16s}{"norm2_diff":16s}')
        for i in range(max_iter):
            norm2_old = norm2
            theta, norm2 = opt.step_and_cost(self.cost_dm,
                                             theta, projector=projector, dm_target=dm_target, shots=shots,
                                             nums_frequency={'theta': {(i,): 1 for i in range(len(theta))}})
            norm2_diff = abs(norm2 - norm2_old)
            print(f'{i+1:<6d}{norm2:<16f}{norm2_diff:<16f}')
            if norm2_diff < tol: break

        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.init_dm.__name__} time elapsed: {tm}m {ts}s', end='\n\n')
        return theta, dm_target
        """

    def dft(self, max_iter=1000, e_tol=1e-3, pat_max=3, mix=0.3):
        t0 = time.time()

        dm = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        pat = 0
        for itr in range(max_iter):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm) # fock = h1e + vhf
            energy, coeff = self.mf.eig(fock, self.s1e)
            occ = self.mf.get_occ(energy, coeff)
            dm = self.mf.make_rdm1(coeff, occ)
            dm = mix * dm + (1 - mix) * dm_last # Mixing
            #vhf = self.mf.get_veff(self.mat, dm, dm_last, vhf) # DIIS
            vhf = self.mf.get_veff(self.mat, dm, dm_last, vhf)
            etot = self.mf.energy_tot(dm, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_last:9f}')

            if abs(etot - etot_last) < e_tol:
                if pat == pat_max: break
                else: pat += 1

        with open(f'{self.output_dir}/dft.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.dft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def qdft(self, theta, shots=None, max_iter=1000, e_tol=1e-6, mix=0.3, alpha=10):
        projector = self.gen_projector()
        theta, dm = self.init_dm(theta, projector, shots)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        t0 = time.time()
        for itr in range(max_iter):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm)
            hamiltonian = qml.Hamiltonian(fock.ravel(), projector)
            res = scipy.optimize.minimize(self.cost_qdft, theta,
                                          args=(projector, shots, fock, hamiltonian, alpha),
                                          method='L-BFGS-B',
                                          options={'disp': False})
            theta = res.x
            dm = mix * self.get_dm(theta) + (1 - mix) * dm_last
            vhf = self.mf.get_veff(self.mat, dm)
            etot = self.mf.energy_tot(dm, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_last:9f}')

            if abs(etot - etot_last) < e_tol: break

        with open(f'{self.output_dir}/qdft.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.qdft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def qudft(self, theta, shots=None, max_iter=1000, tol=1e-4):
        projector = self.gen_projector()
        theta, _ = self.init_dm(theta, projector, shots)

        t0 = time.time()
        res = scipy.optimize.minimize(self.cost_qudft, theta,
                                      args=(projector, shots),
                                      method='L-BFGS-B',
                                      tol=tol,
                                      options={'disp': True})
        theta = res.x
        etot = self.cost_qudft(theta, projector, shots) 

        """
        opt = qml.RotosolveOptimizer()
        etot = 999

        print(f'\n{self.qudft.__name__}\n{"iter":6s}{"etot":16s}{"etot_diff":16s}')
        for i in range(max_iter):
            etot_old = etot
            theta, etot = opt.step_and_cost(self.cost_qudft, theta, 
                                            projector=projector, shots=shots,
                                            nums_frequency={'theta': {(i,): 1 for i in range(len(theta))}})
            etot_diff = abs(etot - etot_old)
            print(f'{i+1:<6d}{etot:<16f}{etot_diff:<16f}')
            if etot_diff < tol: break
        """

        with open(f'{self.output_dir}/qudft.etot', 'w') as f: f.write(f'{etot:f}')
        t1 = time.time() - t0
        tm, ts = divmod(int(t1), 60)
        print(f'{self.qudft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')
        with open(f'log/{self.output_tag}_qudft_L{self.L}.log', 'w') as f: f.write(f'{etot:f} {t1}')

if args.mat == 'H':
    from pyscf import gto, scf # Gaussian-Type Orbital (GTO)
    N = 4
    a = 1.0
    #for a in np.arange(0.5, 2.05, 0.05):
    mat = gto.M(atom='; '.join([f'H {a*i:.1f} 0 0' for i in range(N)]),
                basis='sto-3g',
                pseudo='gth-lda')
    mf = scf.RKS(mat, xc='LDA') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
    output_tag = f'H{N:d}_a{a:.2f}'
elif args.mat == 'Si':
    from pyscf.pbc import gto, scf
    a = 5.46
    mat = gto.M(atom=f'Si 0 0 0; Si {a/4} {a/4} {a/4}',
                a=[[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]],
                basis='gth-szv',
                pseudo='gth-pbe')
    kpts=mat.make_kpts([2, 1, 1])
    mf = scf.RKS(mat, xc='PBE', kpts=kpts)
    output_tag = f'Si_a{a:.2f}'
elif args.mat == 'Cu':
    from pyscf.pbc import gto, scf
    a = 3.58
    mat = gto.M(atom=f'Cu 0 0 0',
                a=[[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]],
                spin=1,
                basis='gth-szv-molopt-sr',
                pseudo='gth-pbe')
    mf = scf.UKS(mat, xc='PBE') # Unrestricted KS-DFT (UKS): the orbitals can have either alpha or beta spin
    #kpts=mat.make_kpts([9, 9, 9])
    #mf = scf.UKS(mat, xc='PBE', kpts=kpts)
    output_tag = f'Cu_a{a:.2f}'
elif args.mat == 'Fe':
    from pyscf.pbc import gto, scf
    a = 2.86
    mat = gto.M(atom=f'Fe 0 0 0',
                a=[[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]],
                spin=4,
                basis='gth-szv-molopt-sr',
                pseudo='gth-pbe')
    mf = scf.UKS(mat, xc='PBE')
    output_tag = f'Fe_a{a:.2f}'

vqe = VQE(mat, mf, output_tag, L=3)
#theta = 2*np.pi * np.random.rand(vqe.nkpts * vqe.M * (vqe.L+1))
theta = 2*np.pi * jnp.random.rand(vqe.nkpts * vqe.M * (vqe.L+1), requires_grad=True) # for qml.optimizer
#vqe.dft()
#vqe.qdft(theta)
vqe.qudft(theta)
