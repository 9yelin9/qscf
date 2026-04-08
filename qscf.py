#!/usr/bin/env python3

import os
import sys
import time
import scipy
import argparse
import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('mat', type=str)
args = parser.parse_args()

class VQE:
    def __init__(self, N, mat, mf, output_dir):
        self.N = N
        self.Nocc = self.N // 2
        self.mat = mat
        self.output_dir = f'output/{output_dir}'
        os.makedirs(self.output_dir, exist_ok=True)

        self.mf = mf
        self.s1e = self.mf.get_ovlp(self.mat)
        self.h1e = self.mf.get_hcore(self.mat)
        self.shp = scipy.linalg.fractional_matrix_power(self.s1e,  0.5)
        self.shm = scipy.linalg.fractional_matrix_power(self.s1e, -0.5)

        self.L = 3
        self.M = int(np.log2(self.N))
        self.wires = list(range(self.M))
        self.bases = np.array([list(np.binary_repr(i, width=self.M)) for i in range(self.N)], dtype='int')
        self.weight = [(self.N - k) / (self.N * (self.N + 1) / 2) for k in range(self.N)]

        self.projector = self.gen_projector()

        #self.dev = qml.device('default.qubit', wires=self.M)
        self.dev = qml.device('lightning.gpu', wires=self.M)
        self.qnode_gamma  = qml.QNode(self.circuit_gamma,  self.dev)
        self.qnode_energy = qml.QNode(self.circuit_energy, self.dev)

    def gen_projector(self):
        X  = np.array([[0,   1], [1,  0]], dtype='complex')
        Y  = np.array([[0, -1j], [1j, 0]], dtype='complex')
        iY = 1j * Y 

        projector = []
        for i, j in [(i, j) for i in self.bases for j in self.bases]:
            p_ij = np.eye(self.N, dtype='complex')

            for mu in self.wires:
                Il = np.eye(2 ** mu)
                Ir = np.eye(2 ** (self.M - mu - 1))

                sign = (i[mu] << 1) - 1
                if i[mu] ^ j[mu]: p = (X - sign * iY) / 2
                else:             p = (X - sign * iY) @ (X + sign * iY) / 4
                p_ij @= np.kron(np.kron(Il, p), Ir)
            p_ij = (p_ij + p_ij.T.conj()) / 2
            projector.append(qml.Hermitian(p_ij, wires=self.wires))

        return projector

    def prep_circuit(self, theta, basis):
        qml.BasisState(basis, wires=self.wires)
        for m in self.wires: qml.RY(theta[m], wires=m)
        for n in range(self.L):
            for m in self.wires[:-1]: qml.CNOT(wires=[m, m+1])
            for m in self.wires:      qml.RY(theta[self.M * (n+1) + m], wires=m)

    def circuit_gamma(self, theta, basis):
        self.prep_circuit(theta, basis)
        return [qml.expval(p) for p in self.projector] #qml.state()

    def circuit_energy(self, theta, basis, hamiltonian):
        self.prep_circuit(theta, basis)
        return qml.expval(hamiltonian)

    def get_dm(self, theta, shots):
        ### TODO: GPU parallerization
        gamma = np.array([self.qnode_gamma(theta, basis, shots=shots) for basis in self.bases[:self.Nocc]]).reshape(-1, self.N, self.N)
        dm = 2 * np.sum([self.shm.conj().T @ (0.5 * (g + g.conj().T)) @ self.shm for g in gamma], axis=0)
        #ne, occ = np.trace(self.s1e @ dm).real, np.linalg.eigvalsh(self.shp @ dm @ self.shp)
        #print(f'ne= {ne:9f} occ= {occ} => {sum(occ)}')
        return dm

    def cost_qdft(self, theta, fock, hamiltonian, alpha, shots):
        e_ks = np.repeat([self.qnode_energy(theta, basis, hamiltonian, shots=shots) for basis in self.bases[:self.Nocc]], 2)
        e_ks_sum = np.sum(np.dot(e_ks, self.weight))

        dm = self.get_dm(theta)
        R = fock @ dm @ self.s1e - self.s1e @ dm @ fock # FDS - SDF = 0
        return e_ks_sum + alpha * np.linalg.norm(R, 'fro')**2

    def cost_qscf(self, theta, shots):
        dm = self.get_dm(theta, shots)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)
        return etot

    def cost_dm(self, theta, dm_target, shots):
        dm = self.get_dm(theta, shots)
        return np.linalg.norm(dm - dm_target)**2

    def init_dm(self, theta, shots):
        dm_target = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)
        res = scipy.optimize.minimize(self.cost_dm, theta,
                                      args=(dm_target, shots),
                                      method='COBYLA',
                                      options={'disp': True})
        return np.array(res.x, dtype=float), dm_target

    def dft(self):
        t0 = time.time()

        self.mf.kernel()

        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.dft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def dft_custom(self, max_iter=1000, e_tol=1e-6, mix=0.3):
        t0 = time.time()

        dm = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        for itr in range(max_iter):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm) # fock = h1e + vhf
            energy, coeff = self.mf.eig(fock, self.s1e)
            occ = self.mf.get_occ(energy, coeff)
            dm = self.mf.make_rdm1(coeff, occ)
            #dm = mix * dm + (1 - mix) * dm_last
            #vhf = self.mf.get_veff(self.mat, dm, dm_last, vhf) # DIIS
            vhf = self.mf.get_veff(self.mat, dm, dm_last, vhf)
            etot = self.mf.energy_tot(dm, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_last:9f}')

            if abs(etot - etot_last) < e_tol: break

        with open(f'{self.output_dir}/dft.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.dft_custom.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def qdft(self, theta, max_iter=1000, e_tol=1e-6, mix=0.3, alpha=10, shots=None):
        theta, dm = self.init_dm(theta, shots)
        vhf = self.mf.get_veff(self.mat, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        t0 = time.time()
        for itr in range(max_iter):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm)
            hamiltonian = qml.Hamiltonian(fock.ravel(), self.projector)
            res = scipy.optimize.minimize(self.cost_qdft, theta,
                                          args=(fock, hamiltonian, alpha, shots),
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

    def qscf(self, theta, max_iter=1000, shots=None):
        theta, _ = self.init_dm(theta, shots)

        t0 = time.time()
        """
        res = scipy.optimize.minimize(self.cost_qscf, theta,
                                      args=(shots,),
                                      method='L-BFGS-B',
                                      options={'disp': True})
        theta = res.x
        etot = self.cost_qscf(theta, shots) 
        """

        etot = 999
        opt = qml.RotosolveOptimizer()
        nums_frequency = {(i,): 1 for i in range(len(theta))}

        print(f'\n{"etot":16s}{"etot_diff":16s}{"shots":6s}')
        for _ in range(max_iter):
            etot_old = etot
            theta, etot = opt.step_and_cost(self.cost_qscf,
                                            theta, shots=shots,
                                            nums_frequency={'theta': nums_frequency})
            etot_diff = abs(etot - etot_old)
            print(f'{etot:<16f}{etot_diff:<16f}{shots:<6d}')

            if etot_diff < 1e-3: break

            shots_new = max(shots_min, min(shots_max, int(sensitive / etot_diff)))
            if shots_new > shots: shots = shots_new

        with open(f'{self.output_dir}/qscf.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.qscf.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

if args.mat == 'H':
    from pyscf import gto, scf # Gaussian-Type Orbital (GTO)
    N = 4
    a = 1.0
    #for a in np.arange(0.5, 2.05, 0.05):
    mat = gto.M(atom='; '.join([f'H {a*i:.1f} 0 0' for i in range(N)]),
                   basis='sto-3g',
                   pseudo='gth-lda')
    mf = scf.RKS(mat, xc='lda') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
    vqe = VQE(N, mat, mf, f'H{N:d}_a{a:.2f}')
elif args.mat == 'Si':
    from pyscf.pbc import gto, scf
    N = 8
    a = 5.459
    mat = gto.M(atom=f'Si 0 0 0; Si {a/4} {a/4} {a/4}',
                   a = [[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]],
                   basis='gth-szv',
                   pseudo='gth-pbe')
    mf = scf.RKS(mat, xc='PBE')
    vqe = VQE(N, mat, mf, f'Si_a{a:.2f}')

theta = 2*np.pi * np.random.rand(vqe.M * (vqe.L+1))
#theta = 2*np.pi * np.random.rand(vqe.M * (vqe.L+1), requires_grad=True) # for qml.optimizer
vqe.dft()
vqe.dft_custom()
#vqe.qdft(theta)
vqe.qscf(theta)
