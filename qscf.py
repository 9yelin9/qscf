#!/usr/bin/env python3

import os
import sys
import time
import scipy
import pennylane as qml
from pennylane import numpy as np
from pyscf import gto, scf

np.random.seed(42)

class VQE:
    def __init__(self, N=8, R=1.0):
        self.N = N
        self.R = R
        self.Nocc = self.N // 2
        self.dir_output = f'output/H{self.N:d}_R{self.R:.2f}'
        os.makedirs(self.dir_output, exist_ok=True)

        self.mol = gto.M(atom = '; '.join([f'H {self.R*i:.1f} 0 0' for i in range(self.N)]), # Gaussian-Type Orbital (GTO)
                     basis = 'sto-3g',
                     pseudo = 'gth-lda')

        self.mf = scf.RKS(self.mol, xc='lda') # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
        self.s1e = self.mf.get_ovlp(self.mol)
        self.h1e = self.mf.get_hcore(self.mol)
        self.shp = scipy.linalg.fractional_matrix_power(self.s1e,  0.5)
        self.shm = scipy.linalg.fractional_matrix_power(self.s1e, -0.5)

        self.L = 3
        self.M = int(np.log2(self.N))
        self.wires = list(range(self.M))
        self.bases = np.array([list(np.binary_repr(i, width=self.M)) for i in range(self.N)], dtype='int')
        self.weight = [(self.N - k) / (self.N * (self.N + 1) / 2) for k in range(self.N)]

        self.maxiter = 1000

        self.projector = self.gen_projector()

        #self.dev = qml.device('default.qubit', wires=self.M)
        self.dev = qml.device('lightning.gpu', wires=self.M)
        #self.dev = qml.device('lightning.gpu', wires=self.M, shots=1000)
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

    def get_dm(self, theta):
        gamma = np.array([self.qnode_gamma(theta, basis) for basis in self.bases[:self.Nocc]]).reshape(-1, self.N, self.N)
        dm = 2 * np.sum([self.shm.conj().T @ (0.5 * (g + g.conj().T)) @ self.shm for g in gamma], axis=0)
        #ne, occ = np.trace(self.s1e @ dm).real, np.linalg.eigvalsh(self.shp @ dm @ self.shp)
        #print(f'ne= {ne:9f} occ= {occ} => {sum(occ)}')
        return dm

    def cost_qdft(self, theta, fock, hamiltonian, alpha):
        e_ks = np.repeat([self.qnode_energy(theta, basis, hamiltonian) for basis in self.bases[:self.Nocc]], 2)
        e_ks_sum = np.sum(np.dot(e_ks, self.weight))

        dm = self.get_dm(theta)
        R = fock @ dm @ self.s1e - self.s1e @ dm @ fock # FDS - SDF = 0
        return e_ks_sum + alpha * np.linalg.norm(R, 'fro')**2

    def cost_qscf(self, theta):
        dm = self.get_dm(theta)
        vhf = self.mf.get_veff(self.mol, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)
        return etot

    def cost_dm(self, theta, dm_target):
        dm = self.get_dm(theta)
        return np.linalg.norm(dm - dm_target)**2

    def init_dm(self, theta):
        dm_target = self.mf.get_init_guess(self.mol, self.mf.init_guess, s1e=self.s1e)
        """
        res = scipy.optimize.minimize(self.cost_dm, theta,
                                      args=(dm_target, ),
                                      method='L-BFGS-B',
                                      options={'disp': False})
        return np.array(res.x, dtype=float), dm_target
        """
        norm2_log = [999] * 10
        opt = qml.SPSAOptimizer(maxiter=self.maxiter)
        for _ in range(self.maxiter):
            norm2_old = norm2
            theta, norm2 = opt.step_and_cost(self.cost_dm, theta, dm_target=dm_target)
            print(f'{norm2:f}, {abs(norm2_old - norm2):f}')
            if abs(abs(norm2_old - norm2)) < 1e-4: break
        return theta, dm_target

    def dft(self):
        t0 = time.time()

        self.mf.kernel()

        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.dft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def dft_custom(self, e_tol=1e-6, mix=0.3):
        t0 = time.time()

        dm = self.mf.get_init_guess(self.mol, self.mf.init_guess, s1e=self.s1e)
        vhf = self.mf.get_veff(self.mol, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        for itr in range(self.max_itr):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm) # fock = h1e + vhf
            energy, coeff = self.mf.eig(fock, self.s1e)
            occ = self.mf.get_occ(energy, coeff)
            dm = mix * self.mf.make_rdm1(coeff, occ) + (1 - mix) * dm_last
            vhf = self.mf.get_veff(self.mol, dm, dm_last, vhf) # DIIS 쓰면 빨라지는 듯?
            etot = self.mf.energy_tot(dm, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_last:9f}')

            if abs(etot - etot_last) < e_tol: break

        with open(f'{self.dir_output}/dft.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.dft_custom.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def qdft(self, theta, e_tol=1e-6, mix=0.3, alpha=10):
        t0 = time.time()

        theta, dm = self.init_dm(theta)
        vhf = self.mf.get_veff(self.mol, dm)
        etot = self.mf.energy_tot(dm, self.h1e, vhf)

        for itr in range(self.max_itr):
            dm_last = dm
            etot_last = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, dm)
            hamiltonian = qml.Hamiltonian(fock.ravel(), self.projector)
            res = scipy.optimize.minimize(self.cost_qdft, theta,
                                          args=(fock, hamiltonian, alpha),
                                          method='L-BFGS-B',
                                          options={'disp': False})
            theta = res.x
            dm = mix * self.get_dm(theta) + (1 - mix) * dm_last
            vhf = self.mf.get_veff(self.mol, dm)
            etot = self.mf.energy_tot(dm, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_last:9f}')

            if abs(etot - etot_last) < e_tol: break

        with open(f'{self.dir_output}/qdft.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.qdft.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

    def qscf(self, theta):
        t0 = time.time()

        theta, _ = self.init_dm(theta)
        print()
        """
        res = scipy.optimize.minimize(self.cost_qscf, theta,
                                      method='L-BFGS-B',
                                      options={'disp': True})
        theta = res.x
        etot = self.cost_qscf(theta) 
        """
        etot = 999
        opt = qml.SPSAOptimizer(maxiter=self.maxiter)
        for _ in range(self.maxiter):
            etot_old = etot
            theta, etot = opt.step_and_cost(self.cost_qscf, theta)
            print(f'{etot:f}, {abs(etot - etot_old)}')
            if abs(etot - etot_old) < 1e-6: break

        with open(f'{self.dir_output}/qscf.etot', 'w') as f: f.write(f'{etot:f}')
        tm, ts = divmod(int(time.time() - t0), 60)
        print(f'{self.qscf.__name__} time elapsed: {tm}m {ts}s', end='\n\n')

#for R in np.arange(0.5, 2.05, 0.05):
vqe = VQE(N=4, R=1.0)
theta = 2*np.pi * np.random.rand(vqe.M * (vqe.L+1), requires_grad=True)
#vqe.dft()
#vqe.dft_custom()
#vqe.qdft(theta)
vqe.qscf(theta)
