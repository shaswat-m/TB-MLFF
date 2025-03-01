import os
import re
import sys

import pyscal.catom as pca
import pyscal.core as pc
from md_util import *
from post_util import *


def read_ase_raw(filename, cn=0):
    """Load ASE data dump

    Parameters
    ----------
    filename : string
        Filename string, should be of '.data' type
    cn: int 0 or 1
        configuration type
    Returns
    -------
    atoms: ASE atoms object
    pos : ndarray
        Real coordiantes of atoms
    box : ndarray
        Simulation box size (c1|c2|c3)
    raw_data : array of size (Natoms,8)
        contains id, type, x, y, z, fx, fy, fz
    """
    if cn == 0:
        posa, hin, raw_data = load_custom_raw(filename)
    elif cn == 1:
        posa, hin, raw_data = load_custom_raw(filename, 2, 5, 15)
    h = hin[:, 1] - hin[:, 0]  # /3.405
    pos = posa.copy()  # /3.405
    box = np.array([[h[0], 0.0, 0.0], [0.0, h[1], 0.0], [0.0, 0.0, h[2]]])
    atoms = []
    for i in range(pos.shape[0]):
        atom = pca.Atom()
        atom.pos = list(pos[i, :])
        atoms.append(atom)

    return atoms, box, pos, raw_data


def read_ase(filename):
    """Load ASE data into ASE Atoms object

    Parameters
    ----------
    filename : string
        Filename string, should be of '.data' type
    Returns
    -------
    atoms: ASE atoms object
    box : ndarray
        Simulation box size (c1|c2|c3)
    """
    posa, hin = load_custom(filename)
    h = (hin[:, 1] - hin[:, 0]) / 3.405
    pos = posa.copy() / 3.405
    box = np.array([[h[0], 0.0, 0.0], [0.0, h[1], 0.0], [0.0, 0.0, h[2]]])
    pos = pbc(pos, box)
    atoms = []
    for i in range(pos.shape[0]):
        atom = pca.Atom()
        atom.pos = list(pos[i, :])
        atoms.append(atom)

    return atoms, box


def read_lammps(posa, hin):
    """Load LAMMPS data file into ASE object

    Parameters
    ----------
    pos_a : array of shape (Natoms,3)
         Array of atomic positions
    hin: array of shape (3,2)
        box limits in x, y, and z direction
    Returns
    -------
    atoms: ASE atoms object
    box : ndarray
        Simulation box size (c1|c2|c3)
    """
    h = hin[:, 1] - hin[:, 0]  # /3.405
    pos = posa.copy()  # /3.405
    box = np.array([[h[0], 0.0, 0.0], [0.0, h[1], 0.0], [0.0, 0.0, h[2]]])
    pos = pbc(pos, box)
    atoms = []
    for i in range(pos.shape[0]):
        atom = pca.Atom()
        atom.pos = list(pos[i, :])
        atoms.append(atom)

    return atoms, box


def fraction_liquid_ase(filename, natoms=None):
    """Fraction of atoms in liquid state while reading ASE input file

    Parameters
    ----------
    filename : string
        Filename string, should be of '.data' type
    natoms: int
        Total number of atoms
    Returns
    -------
    frac: float
        Fraction of atoms in liquid state
    """
    ps = pc.System()
    ps.atoms, ps.box = read_ase(filename)
    ps.find_neighbors(method="cutoff", cutoff=1.4)
    nn = [atom.coordination for atom in ps.atoms]
    nn = np.array(nn)
    frac = len(np.where(nn != 12)[0]) / natoms
    return frac


def fraction_liquid_lammps(pos, hi, natoms=None):
    """Fraction of atoms in liquid state while reading LAMMPS input file

    Parameters
    ----------
    pos_a : array of shape (Natoms,3)
         Array of atomic positions
    hin: array of shape (3,2)
        box limits in x, y, and z direction
    natoms: int
        Total number of atoms
    Returns
    -------
    frac: float
        Fraction of atoms in liquid state
    """
    ps = pc.System()
    ps.atoms, ps.box = read_lammps(pos, hi)
    ps.find_neighbors(method="cutoff", cutoff=1.4)
    nn = [atom.coordination for atom in ps.atoms]
    nn = np.array(nn)
    frac = len(np.where(nn != 12)[0]) / natoms
    return frac


def perturb_and_write(filename, cn=0):
    """Load ASE data dump

    Parameters
    ----------
    filename : string
        Filename string, should be of '.data' type
    cn: int 0 or 1
        configuration type
    Returns
    -------
    COM: array of size (3,)
    pos : array of size (Natoms,3)
        Real coordinates of atoms
    pos_vac : position of vacancy
    """
    psys = pc.System()
    psys.atoms, psys.box, pos, raw = read_ase_raw(filename, cn=cn)
    psys.find_neighbors(method="voronoi")
    CN = np.array(psys.calculate_centrosymmetry())
    inds = np.linspace(0, len(CN) - 1, len(CN), dtype=int)
    ind = inds[np.argsort(CN)]
    ind = ind[-12:]
    pos_temp = pos[ind, :].copy()
    mask = pos_temp.max(0) - pos_temp.min(0) > psys.box[0][0] / 2.0
    if True in mask:
        pos_temp[np.where(pos_temp < psys.box[0][0] / 2.0)] += psys.box[0][0]
        pos[ind, :] = pos_temp.copy()
    COM = pos_temp.mean(0)
    pind = np.random.choice(ind, 1)
    vec = COM - pos[pind, :]

    lines = open(filename, "r").readlines()
    if cn == 0:
        pos[pind, :] += 0.6 + 0.5 * np.random.rand(1) * vec
        raw[pind, 2:5] = pos[pind, :].copy()
        fid = open(filename[:12] + "temp" + filename[14:], "w")
        for i in range(9):
            fid.write(lines[i])
        for i in range(pos.shape[0]):
            fid.write(
                "%d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"
                % (
                    int(raw[i, 0]),
                    int(raw[i, 1]),
                    raw[i, 2],
                    raw[i, 3],
                    raw[i, 4],
                    raw[i, 5],
                    raw[i, 6],
                    raw[i, 7],
                    raw[i, 8],
                    raw[i, 9],
                    raw[i, 10],
                )
            )
        fid.close()
    if cn == 1:
        fid = open("final.neb", "w")
        fid.write("1\n")
        fid.write("%d %f %f %f" % (raw[pind, 0], COM[0], COM[1], COM[2]))
        fid.close()
        return COM, pos[ind, :], pos[pind, :]


def fraction_liquid(ps, natoms):
    nn = [atom.coordination for atom in ps.atoms]
    nn = np.array(nn)
    frac = len(np.where(nn != 12)[0]) / natoms
    return frac
