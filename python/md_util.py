# --------------------------------------------------------
# md_util.py.py
# by Shaswat Mohanty, shaswatm@stanford.edu
# last modified : Sat Oct 29 2022 12:13:23 2022

# --------------------------------------------------------

import os
import numpy as np
from itertools import combinations
import re
#from numba import jit, prange
from scipy.signal import savgol_filter
from scipy.special import erf
#from numba import set_num_threads
import argparse
import itertools
from functools import partial
import multiprocessing as mp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def objective(y,a):
    return a*y

def objective_lin(x,a,b):
    return a*x+b
######################## io functions ########################
def load_atom_data(filename, skip_to_np, skip_to_box, skip_to_pos, verbose=True, style='full', h_full=True):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS data file', filename)

    nparticles = np.genfromtxt(filename, skip_header=skip_to_np,  dtype=np.int,    max_rows=1).item(0)
    box        = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:,:2]
    if h_full:
        h = np.diag(box[:,1]-box[:,0])
    else:
        h=box
    rawdata    = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata=rawdata[np.argsort(rawdata[:,0])]
    pos = np.zeros([nparticles, 3])
    if style=='full':
        pos[rawdata[:,0].astype(int)-1] = rawdata[:,4:7]
    elif style=='bond':
        pos[rawdata[:,0].astype(int)-1] = rawdata[:,3:6]

    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    return pos, h

def load_atom_data_binary(filename, skip_to_np, skip_to_box, skip_to_pos, verbose=True, style='full',family=None,h_full=True,fill=0, atom_ids = False):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS data file', filename)

    nparticles = np.genfromtxt(filename, skip_header=skip_to_np,  dtype=np.int,    max_rows=1).item(0)
    box        = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:,:2]
    if h_full:
        h = np.diag(box[:,1]-box[:,0])
    else:
        h=box
    rawdata    = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata=rawdata[np.argsort(rawdata[:,0])]
    o_ct=0
    if family=='polymer':
        o_ct=1+fill
    if style == 'sphere':
        a_col = 1
    else:
        a_col = 2
    ind_a=np.where(abs(rawdata[:,a_col]-(1+o_ct))<1e-3)
    ind_b=np.where(abs(rawdata[:,a_col]-(2+o_ct))<1e-3)
    atoms_a=rawdata[ind_a,0].shape[1]
    atoms_b=rawdata[ind_b,0].shape[1]
    pos_a = np.zeros((atoms_a, 3))
    pos_b = np.zeros((atoms_b, 3))
    if style=='full':
        pos_a[:,:] = rawdata[ind_a,4:7]
        pos_b[:,:] = rawdata[ind_b,4:7]
    elif style=='bond':
        pos_a[:,:] = rawdata[ind_a,3:6]
        pos_b[:,:] = rawdata[ind_b,3:6]
    if atom_ids:
        ind_tot = []
        ind_tot.append(ind_b)
        ind_tot.append(ind_a)
        id_no = rawdata[ind_tot,0]
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    if atom_ids:
        return pos_a, pos_b, h, atoms_a, atoms_b, id_no
    else:
        return pos_a, pos_b, h, atoms_a, atoms_b

def load_dumpfile_atom_data(filename, total_steps, dump_frequency, verbose=True, at_type=1, h_full=True, add = 'position'):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np  = 3
    fid=open(filename,'r')
    lines=fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()),dtype=int)
    nparticles=npar[0]
    ndum=int(total_steps/dump_frequency)
    if h_full:
        h=np.zeros([ndum,3])
    else:
        h=np.zeros([3,2,ndum])
    pos = np.zeros([ndum*nparticles, 3])
    if at_type==1:
        ct=1
    elif at_type==2:
        ct=0
    for i in range(ndum):
        skip_to_pos=(i+ct)*nparticles+(i+ct+1)*9
        skip_to_box=(i+ct)*nparticles+(i+ct+1)*9-4
        if h_full:	
            h[i,:] = get_h_from_lines(lines, skip_to_box) 
        else:   
            h[:,:,i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata    = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata=rawdata[np.argsort(rawdata[:,0])]
        pos[i*nparticles:(i+1)*nparticles] = rawdata[:,2:5]
        if h_full:
            pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(h[i,:])) 
        else:
            hin=h[:,1,i]-h[:,0,i]
            pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(hin)) 
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
#    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_atom_data_fast(filename, total_steps, dump_frequency, verbose=True, h_full=True, at_type=1, add='position'):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np  = 3
    fid=open(filename,'r')
    lines=fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()),dtype=int)
    nparticles=npar[0]
    ndum=int(total_steps/dump_frequency)
    if h_full:
        h=np.zeros([ndum,3])
    else:
        h=np.zeros([3,2,ndum])
    pos = np.zeros([int(ndum*nparticles), 3])
    if at_type==1:
        ct=1
    elif at_type==2:
        ct=0
    for i in range(ndum):
        skip_to_pos=(i+ct)*nparticles+(i+ct+1)*9
        skip_to_box=(i+ct)*nparticles+(i+ct+1)*9-4
        if h_full:	
            h[i,:] = get_h_from_lines(lines, skip_to_box) 
        else:   
            h[:,:,i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata    = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata=rawdata[np.argsort(rawdata[:,0])]
        pos[i*nparticles:(i+1)*nparticles] = rawdata[:,2:5]
        if h_full:
            pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(h[i,:])) 
        else:
            hin=h[:,1,i]-h[:,0,i]
            pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(hin)) 
 
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
#    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_velocity(filename, total_steps, dump_frequency, verbose=True, at_type=1):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np  = 3
    fid=open(filename,'r')
    lines=fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()),dtype=int)
    nparticles=npar[0]
    ndum=int(total_steps/dump_frequency)
    h=np.zeros([ndum,3])
    pos = np.zeros([int(ndum*nparticles), 3])
    if at_type==1:
        ct=1
    elif at_type==2:
        ct=0
    for i in range(ndum):
        skip_to_pos=(i+ct)*nparticles+(i+ct+1)*9
        skip_to_box=(i+ct)*nparticles+(i+ct+1)*9-4
        h[i,:] = get_h_from_lines(lines, skip_to_box) 
        rawdata    = get_pos_from_lines(lines, skip_to_pos, nparticles, add='velocity')
        rawdata=rawdata[np.argsort(rawdata[:,0])]
        pos[i*nparticles:(i+1)*nparticles] = rawdata[:,5:]
#        pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(h[i,:])) 
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
#    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_atom_data_binary(filename, total_steps, dump_frequency, verbose=True, at_type=1,family=None,fill=0):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np  = 3
    nparticles = np.genfromtxt(filename, skip_header=skip_to_np,  dtype=np.int,    max_rows=1).item(0)
    ndum=int(total_steps/dump_frequency)
    h=np.zeros([ndum,3])
    pos = np.zeros([ndum*nparticles, 3])
    pos_n = np.zeros([ndum*nparticles, 5])
    o_ct=0
    if family=='polymer':
        o_ct=1+fill
    
    if at_type==1:
        ct=1
    elif at_type==2:
        ct=0
    for i in range(ndum):
        skip_to_pos=(i+ct)*nparticles+(i+ct+1)*9
        skip_to_box=(i+ct)*nparticles+(i+ct+1)*9-4
        box = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:,:2]
        h[i,:] = np.transpose(box[:,1]-box[:,0])
        rawdata    = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
        rawdata=rawdata[np.argsort(rawdata[:,0])]
        pos[i*nparticles:(i+1)*nparticles] = rawdata[:,2:5]
        pos_n[i*nparticles:(i+1)*nparticles] = rawdata[:,:5]
        pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(h[i,:])) 
    ind_a=np.where(abs(pos_n[:,1]-(o_ct+1))<1e-3)
    ind_b=np.where(abs(pos_n[:,1]-(o_ct+2))<1e-3)
    atoms_a=int(pos[ind_a,0].shape[1]/ndum)
    atoms_b=int(pos[ind_b,0].shape[1]/ndum)
    pos_a = pos[ind_a,:]
    pos_b = pos[ind_b,:]
    pos_a=pos_a.reshape(-1,3)
    pos_b=pos_b.reshape(-1,3)
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
#    pos=np.dot(pos,h)
    return pos_a, pos_b, h, atoms_a, atoms_b

def load_dumpfile_atom_data_binary_fast(filename, total_steps, dump_frequency, h_full=True, verbose=True, at_type=1,add='position',family=None,fill=0):
    '''Load LAMMPS .data type configuration files
    
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information
        
        Returns
        -------
        pos : ndarray
            Real coordiantes of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))
    
    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np  = 3
    fid=open(filename,'r')
    lines=fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()),dtype=int)
    nparticles=npar[0]
    ndum=int(total_steps/dump_frequency)
    if h_full:
        h=np.zeros([ndum,3])
    else:
        h=np.zeros([3,2,ndum])
    pos = np.zeros([ndum*nparticles, 3])
    pos_n = np.zeros([ndum*nparticles, 5])
    o_ct=0
    if family=='polymer':
        o_ct=1+fill
    
    if at_type==1:
        ct=1
    elif at_type==2:
        ct=0
    for i in range(ndum):
        skip_to_pos=(i+ct)*nparticles+(i+ct+1)*9
        skip_to_box=(i+ct)*nparticles+(i+ct+1)*9-4
        if h_full:	
            h[i,:] = get_h_from_lines(lines, skip_to_box) 
        else:   
            h[:,:,i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata    = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata=rawdata[np.argsort(rawdata[:,0])]
        pos[i*nparticles:(i+1)*nparticles,:] = rawdata[:,2:5]
        pos_n[i*nparticles:(i+1)*nparticles,:] = rawdata[:,:5]
        if h_full:
            pos[i*nparticles:(i+1)*nparticles,:] = np.dot(pos[i*nparticles:(i+1)*nparticles,:],np.diag(h[i,:])) 
        else:
            hin=h[:,1,i]-h[:,0,i]
            pos[i*nparticles:(i+1)*nparticles,:] = np.dot(pos[i*nparticles:(i+1)*nparticles,:],np.diag(hin)) 
    ind_a=np.where(abs(pos_n[:,1]-(1+o_ct))<1e-3)
    ind_b=np.where(abs(pos_n[:,1]-(2+o_ct))<1e-3)
    atoms_a=int(pos[ind_a,0].shape[1]/ndum)
    atoms_b=int(pos[ind_b,0].shape[1]/ndum)
    pos_a = pos[ind_a,:]
    pos_b = pos[ind_b,:]
    pos_a=pos_a.reshape(-1,3)
    pos_b=pos_b.reshape(-1,3)
    if verbose:
        print('Nparticles = %d'%(nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
#    pos=np.dot(pos,h)
    return pos_a, pos_b, h, atoms_a, atoms_b

################### boundary conditions ######################
def calculate_strains(h, load = True):
    '''calculate strains in the box

            Considering periodic boundary conditions (PBC)

            Parameters
            ----------
            h : float, dimension (N, 3)
                Box length for N timesteps in x, y, and z directions
            load : boolean
                either during loading or unloading half of the cycle

            Returns
            -------
            ex, ey, ez : float, dimension (N,)
                Strains in all 3 directions

    '''
    if load:
        ex = h[:,0]/h[0,0]-1.0
        ey = h[:,1]/h[0,1]-1.0
        ez = h[:,2]/h[0,2]-1.0
    else:
        ex = h[:,0]/h[-1,0]-1.0
        ey = h[:,1]/h[-1,1]-1.0
        ez = h[:,2]/h[-1,2]-1.0
    return ex, ey, ez

def pbc(drij, h, hinv=None):
    '''calculate distance vector between i and j
    
        Considering periodic boundary conditions (PBC)
    
        Parameters
        ----------
        drij : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        hinv : optional, float, dimension (3, 3)
            inverse matrix of h, if None, it will be calculated

        Returns
        -------
        drij : float, dimension (npairs, 3)
            modified distance vectors of atom pairs considering PBC (Angstrom)
            
    '''
    # Check the input
    if len(drij.shape) == 1:         # Only one pair
        drij = drij.reshape(1, -1)
    if (len(drij.shape) != 2):
        raise ValueError('pbc: drij shape not correct, must be (npairs, nd), (nd = 2,3)')
    npairs, nd = drij.shape 
    if len(h.shape) != 2 or h.shape[0] != h.shape[1] or nd != h.shape[0]:
        raise ValueError('pbc: h matrix shape not consistent with drij')
    # Calculate inverse matrix of h
    if hinv is None:
        hinv = np.linalg.inv(h)

    dsij = np.dot(hinv, drij.T).T
    dsij = dsij - np.round(dsij)
    drij = np.dot(h, dsij.T).T
    
    return drij

def pbc_msd(pos_new, pos_old, h):
    '''calculate the offset vector to add to avoid periodic boundary jump
    
        Considering periodic boundary conditions (PBC)
    
        Parameters
        ----------
        pos_new : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        pos_old : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)

        Returns
        -------
        off_vec : float, dimension (npairs, 3)
            modified distance vectors of atom pairs considering PBC (Angstrom)
            
    '''
    # Check the input
    if len(pos_new.shape) == 1:         # Only one pair
        npairs = 1 
        nd = 3
        off_vec = np.zeros(3)
    else:
        npairs, nd = pos_new.shape
        off_vec = np.zeros((npairs,nd))
    h_f = np.diag(h)
    if npairs == 1:
        diff = pos_new-pos_old
        siff = np.rint(diff/h_f)
        off_vec = -siff*h_f
        #print(diff, siff, off_vec)
        '''
    ind_plus = np.where(diff<-h_f/2.0)[0]
        ind_minus = np.where(diff>h_f/2.0)[0]
        if len(ind_plus)>0:
           off_vec[ind_plus] = h_f[ind_plus]
        if len(ind_minus)>0:
           off_vec[ind_minus] = -h_f[ind_minus]
        '''
    else:
        for i in range(npairs):
            diff = pos_new[i,:]-pos_old[i,:]
            siff = np.rint(diff/h_f)
            off_vec[i,:] = -siff*h_f
            '''
            ind_plus = np.where(diff<-h_f/2.0)[0]
            ind_minus = np.where(diff>h_f/2.0)[0]
            if len(ind_plus)>0:
                off_vec[i,ind_plus] = h_f[ind_plus]
            if len(ind_minus)>0:
                off_vec[i,ind_minus] = -h_f[ind_minus]
            '''
    
    return off_vec
################### neighbor lists ######################

def celllist(r, h, Ns, Ny=None, Nz=None):
    '''Construct cell list in 3D
    
        This function takes the **real coordinates** of atoms `r` and the
        simulation box size `h`. Grouping atoms into Nx x Ny x Nz cells
        
        Parameters
        ----------
        r : float, dimension (nparticles, nd)
            *real* coordinate of atoms
        h : float, dimension (nd, nd)
            Periodic box size h = (c1|c2|c3)
        Ns : tuple, dimension (nd, )
            number of cells in x, y, z direction
        Ny : int
            if not None, represent number of cells in y direction, use with Nx = Ns
        Nz : int
            if not None, represent number of cells in z direction
            
        Returns
        -------
        cell : list, dimension (Nx, Ny, Nz)
            each element cell[i][j][k] is also a list recording all 
            the indices of atoms within the cell[i][j][k].
            (0 <= i < Nx, 0 <= j < Ny, 0 <= k < Nz)
        cellid : int, dimension (nparticles, nd)
            for atom i:
            ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
            atom i belongs to cell[ix][iy][iz]

    '''
    if Ny is not None:
        if Nz is None:
            Ns = (Ns, Ny)
        else:
            Ns = (Ns, Ny, Nz)

    nparticle, nd = r.shape
    if nd != 3 or len(Ns) != 3:
        raise TypeError('celllist: only support 3d cell')

    # create empty cell list of size Nx x Ny x Nz
    cell = np.empty(Ns, dtype=object)
    for i, v in np.ndenumerate(cell):
        cell[i] = []

    # find reduced coordinates of all atoms
    s = np.dot(np.linalg.inv(h), r.T).T
    # fold reduced coordinates into [0, 1) as scaled coordinates
    s = s - np.floor(s)

    # create cell list and cell id list
    cellid = np.floor(s*np.array(Ns)[np.newaxis, :]).astype(np.int)
    for i in range(nparticle):
        cell[tuple(cellid[i, :])].append(i)

    return cell, cellid

def verletlist(r, h, rv, atoms = None, near_neigh = None):
    '''Construct Verlet List (neighbor list) in 3D (vectorized)
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    nparticles, nd = r.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')
    if atoms is not None:
        nparticles = atoms
    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int)
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 100:
        Nx = (Nx/20).astype(np.int)
    if Ny > 100:
        Ny = (Ny/20).astype(np.int)
    if Nz > 100:
        Nz = (Nz/20).astype(np.int)
    
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")

    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell, cellid = celllist(r, h, Nx, Ny, Nz)
    
    # initialize Verlet list
    nn = np.zeros(nparticles, dtype=int)
    nindex = [[] for i in range(nparticles)]
    if near_neigh is not None:
        global_nbr = []
    for i in range(nparticles):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        if near_neigh is not None:
            nbr_inds = []
            nbr_dist = []
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell[nnx][nny][nnz].copy()
                    nc = len(ind)

                    # vectorized implementation
                    if i in ind:
                        ind.remove(i)
                    rj = r[ind,:]
                    drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)

                    ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                    if near_neigh is None:
                        if len(ind_nbrs) > 0:
                            nn[i] += len(ind_nbrs)
                            nindex[i].extend([ind[j] for j in ind_nbrs])
                    else:
                        if len(ind_nbrs) > 0:
                            nbr_inds.extend([ind[j] for j in ind_nbrs])
                            nbr_dist.extend(np.linalg.norm(drij[ind_nbrs], axis=1).tolist())
        if near_neigh is not None:
            if len(nbr_dist)>=near_neigh:
                nbr_dist = np.array(nbr_dist)
                nbr_inds = np.array(nbr_inds)
                nbr_inds = nbr_inds[np.argsort(nbr_dist)].tolist()
                to_add = []
                ct = 0
                for ll in range(len(nbr_inds)):
                    if (nbr_inds[ll] not in global_nbr) and ct < near_neigh:
                        to_add.append(nbr_inds[ll])
                        global_nbr.append(nbr_inds[ll])
                        ct += 1
                if ct == near_neigh:
                    nn[i] = near_neigh
                    nindex[i].append(to_add)

    return nn, nindex

def verletlist_binary(r, rr, h, rv, vectorization = True, near_neigh = None):
    '''Construct Verlet List (neighbor list) in 3D (vectorized)
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    np_a, nd = r.shape
    np_b, nd = rr.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')

    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int) 
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 100:
        Nx = (Nx/20).astype(np.int)
    if Ny > 100:
        Ny = (Ny/20).astype(np.int)
    if Nz > 100:
        Nz = (Nz/20).astype(np.int)
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")
    
    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell_a, cellid_a = celllist(r, h, Nx, Ny, Nz)
    cell_b, cellid_b = celllist(rr, h, Nx, Ny, Nz)
    
    # Find the number of atoms
    np_a = r.shape[0]
    np_b = rr.shape[0]
    
    # initialize Verlet list
    nn_a = np.zeros(np_a, dtype=int)
    nindex_a = [[] for i in range(np_a)]
    nn_b = np.zeros(np_b, dtype=int)
    nindex_b = [[] for i in range(np_b)]
    if near_neigh is not None:
        global_nbr = []
    for i in range(np_a):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        if near_neigh is not None:
            nbr_inds = []
            nbr_dist = []
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid_a[i, 0], cellid_a[i, 1], cellid_a[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell_b[nnx][nny][nnz].copy()
                    nc = len(ind)
                    if vectorization:			
                    # vectorized implementation
                        if i in ind:
                            ind.remove(i)
                        rj = rr[ind,:]
                        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
                        ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                        if near_neigh is None:
                            if len(ind_nbrs) > 0:
                                nn_a[i] += len(ind_nbrs)
                                nindex_a[i].extend([ind[j] for j in ind_nbrs])
                        else:
                            if len(ind_nbrs) > 0:
                                nbr_inds.extend([ind[j] for j in ind_nbrs])
                                nbr_dist.extend(np.linalg.norm(drij[ind_nbrs],axis=1).tolist()) 
 
                    else:
                        for k in range(nc):
                            j = ind[k]
                            # update nn[i] and nindex[i]
                            if i == j:
                                continue
                            else:
                                rj = rr[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                                drij = pbc(rj - ri, h, hinv)
                                if near_neigh is None:
                                    if np.linalg.norm(drij) < rv:
                                        nn_a[i] += 1
                                        nindex_a[i].append(j)
                                else:
                                    if np.linalg.norm(drij) < rv:
                                        nbr_dist.extend(np.linalg.norm(drij).reshape(-1,))
                                        nbr_inds.append(j)
        
        if near_neigh is not None:
            if len(nbr_dist)>=near_neigh:
                nbr_dist = np.array(nbr_dist)
                nbr_inds = np.array(nbr_inds)
                nbr_inds = nbr_inds[np.argsort(nbr_dist)].tolist()
                to_add = []
                ct = 0 
                for ll in range(len(nbr_inds)):
                    if (nbr_inds[ll] not in global_nbr) and ct < near_neigh:
                        to_add.append(nbr_inds[ll])
                        global_nbr.append(nbr_inds[ll])
                        ct += 1
                if ct == near_neigh:
                    nn_a[i] = near_neigh
                    nindex_a[i].append(to_add)

    for i in range(np_b):
        # position of atom i
        ri = rr[i, :].reshape(1, 3)
        
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid_b[i, 0], cellid_b[i, 1], cellid_b[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell_a[nnx][nny][nnz].copy()
                    nc = len(ind)
                    if vectorization:
                    # vectorized implementation
                        if i in ind:
                            ind.remove(i)
                        rj = r[ind,:]
                        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
  
                        ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                        if len(ind_nbrs) > 0:
                            nn_b[i] += len(ind_nbrs)
                            nindex_b[i].extend([ind[j] for j in ind_nbrs])
                    else:
                        for k in range(nc):
                            j = ind[k]
                            # update nn[i] and nindex[i]
                            if i == j:
                                continue
                            else:
                                rj = r[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                                drij = pbc(rj - ri, h, hinv)

                                if np.linalg.norm(drij) < rv:
                                    nn_b[i] += 1
                                    nindex_b[i].append(j)

    return nn_a, nindex_a, nn_b, nindex_b

def bond_vector_binary_par(pos, natom_a, rv, vectorization = False):
    pos_a = pos[:natom_a,:]
    pos_b = pos[natom_a:-1,:]
    h = np.diag(pos[-1,:])
    nn_a, nindex_a, nn_b, nindex_b = verletlist_binary(pos_a, pos_b, h=h, rv=rv, vectorization = vectorization)
    bond_vec = np.zeros(natom_a)
    bond_vec[np.where(nn_a==2)[0]]=1
    return bond_vec

def verletlist_old(r, h, rv):
    '''Construct Verlet List (neighbor list) in 3D
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    nparticle, nd = r.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')

    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int)
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 100:
        Nx = (Nx/20).astype(np.int)
    if Ny > 100:
        Ny = (Ny/20).astype(np.int)
    if Nz > 100:
        Nz = (Nz/20).astype(np.int)
    
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")

    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell, cellid = celllist(r, h, Nx, Ny, Nz)
    
    # Find the number of atoms
    nparticles = r.shape[0]
    
    # initialize Verlet list
    nn = np.zeros(nparticles, dtype=int)
    nindex = [[] for i in range(nparticles)]
    
    for i in range(nparticles):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell[nnx][nny][nnz]
                    nc = len(ind)

                    # go through all the atoms in the neighboring cells
                    for k in range(nc):
                        j = ind[k]
                        # update nn[i] and nindex[i]
                        if i == j:
                            continue
                        else:
                            rj = r[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                            drij = pbc(rj - ri, h, hinv)

                            if np.linalg.norm(drij) < rv:
                                nn[i] += 1
                                nindex[i].append(j)

    return nn, nindex

def convert_nindex_to_array(nn, nindex):
    '''convert the indexes of neighbors to an array

            Parameters
            ----------
            nn : int, dimension (natoms)
                List of number of nearest neighbors per atom
            nindex : int, dimension (natoms)
                List of index of neighbors

            Returns
            -------
            index_array : float, dimension (natoms, nn.max())
                nindex converted to an array

    '''
    nparticles = nn.shape[0]
    index_array = np.ones([nparticles, nn.max().astype(int)], dtype=int)*(-1)
    for i in range(nparticles):
        index_array[i,:nn[i].astype(int)] = nindex[i][:nn[i].astype(int)]
    return index_array

def compute_angle(v1,v2):
    return 180.0-(np.arccos(np.sum(v1*v2)/(v1**2).sum()**0.5/(v2**2).sum()**0.5))*180/np.pi

def compute_angle_distribution(filename,rv=1.1):
    skip_to_np  = 2
    skip_to_box = 7
    skip_to_pos = 29
    pos_a, pos_b, h, atoms_a, atoms_b = load_atom_data_binary(filename, skip_to_np, skip_to_box, skip_to_pos, verbose=False, family='polymer',h_full=False)
    hin = np.diag(h[:,1]-h[:,0])
    nn_a, nindex_a, nn_b, nindex_b = verletlist_binary(pos_a, pos_b, h=hin, rv=rv)

    valid_ind = np.where(nn_a==2)[0]
    centers = len(valid_ind)
    angles = np.zeros(centers)
    for i in range(centers):
        vec1 = pos_b[nindex_a[valid_ind[i]][0]] - pos_a[valid_ind[i]]
        vec2 = pos_b[nindex_a[valid_ind[i]][1]] - pos_a[valid_ind[i]]
        angles[i] = compute_angle(vec1, vec2)
    return angles

def compute_distribution_outline(dataset, num_bins = 100):
    n, bin = np.histogram(dataset, num_bins, density = True)
    bins = 0.5*(bin[1:]+bin[:-1])
    return n, bins

def g_r_verlet(pos, bins, rc, h, a = None, nnlist=None, bin_range = None):
    '''RDF computing function

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            bins : int
                Number of bins
            bin_range : float, dimension(bin,)
                specific bin range instead of bins
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            bin_centers : float, dimension (bins,)
                Centers of all bins
            hist_normalized : float, dimension (bins,)
                Normalized histograms to be equivalent to the g(r)
    '''
    if rc>h.max()/2:
        newpos,newh = config_repeater(pos,h)
        pos = newpos.copy()
        h = newh.copy()
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i,:]])
        ind = index[i]
        rj = pos[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())
    if bin_range is None:
        hist, bin_edges = np.histogram(dist, bins)
    else:
        hist, bin_edges = np.histogram(dist, bins, range=(bin_range[0], bin_range[1]))
    # print("bin_edges = [%g : %g : %g]"%(bin_edges[0], bin_edges[1]-bin_edges[0], bin_edges[-1]))
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    cnt = (4.0/3.0)*np.pi*(np.power(bin_edges[1:],3) - np.power(bin_edges[:-1],3)) * nparticles / np.linalg.det(h)
    hist_normalized = np.divide( hist, cnt ) / nparticles

    return bin_centers, hist_normalized

def g_r_verlet_binary(pos_a, pos_b, bins, rc, a, h):
    '''Partial RDF computing function between A-B types

        Parameters
        ----------
        pos_a : float, dimension (natoms_a, 3)
            Position of all atoms of type A
        pos_b : float, dimension (natoms_b, 3)
            Position of all atoms of type B
        rc : float,
            Cutoff radius for g(r)
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        bins : int
            Number of bins

        Returns
        -------
        bin_centers : float, dimension (bins,)
            Centers of all bins
        hist_normalized : float, dimension (bins,)
            Normalized histograms to be equivalent to the g(r)
    '''
    nn_a, index_a, nn_b, index_b = verletlist_binary(pos_a, pos_b, h, rc)

    atoms_a = pos_a.shape[0]
    atoms_b = pos_b.shape[0]
    nparticles=atoms_a+atoms_b
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(atoms_a):
        ri = np.array([pos_a[i,:]])
        ind = index_a[i]
        rj = pos_b[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    for i in range(atoms_b):
        ri = np.array([pos_b[i,:]])
        ind = index_b[i]
        rj = pos_a[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    hist, bin_edges = np.histogram(dist, bins)
    print("bin_edges = [%g : %g : %g]"%(bin_edges[0], bin_edges[1]-bin_edges[0], bin_edges[-1]))
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    cnt = (4.0/3.0)*np.pi*(np.power(bin_edges[1:],3) - np.power(bin_edges[:-1],3)) * nparticles / np.linalg.det(h)
    hist_normalized = 2.0*np.divide( hist, cnt ) / nparticles

    return bin_centers, hist_normalized

def s_q_from_g_r(q_array, r_array, g_array, r0, rho):
    '''Numerically integrate the g(r) (Fourier Transform) to obtain s(q)

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            r0 : float,
                Lowest value in r-space, where analytical integral is to be computed
            g_array : float, dimension (bins, )
                g(r) along r_array
            rho : float,
                Atomic density

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        r_sinqr = np.multiply(r_array, np.sin(q*r_array))
        int_0_r0 = (r0*q*np.cos(r0*q) - np.sin(r0*q))/np.power(q,3)  # integral from 0 to r0
        s_array[i] = 1.0 + 4.0*np.pi*rho*(int_0_r0 + np.trapz(np.multiply(g_array - 1.0, r_sinqr), x=r_array) / q )
    return s_array

def s_q_from_pos(q_array, pos, h, rc, rho, nnlist=None):
    '''Obtain s(q) from pointwise computation of Fourier transform of atomic positions

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            rho : float,
                Atomic density
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i,:]])
        ind = index[i]
        rj = pos[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    r_array = np.array(dist)
#    r_array= f_weight_vec(r_array,rc)
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        sinqr_div_r = np.divide(np.sin(q*r_array), r_array)#np.exp(-1j * r_array* q)
        int_0_rc = (rc*q*np.cos(rc*q) - np.sin(rc*q))/np.power(q,3)  # integral from 0 to rc
        s_array[i] = 1.0 + 4.0*np.pi*rho*int_0_rc +  np.sum(sinqr_div_r)/(nparticles*q)#np.sum(sinqr_div_r)/(nparticles)
    return s_array.real

def get_q3(k0, ky_relative, kz_relative):
    '''Get a stacked 2D of kx, ky, and kz values on a detector grid

            Parameters
            ----------
            k_0 : float,
                Wave-vector magnitude
            ky_relative : float,
                Non-dimensional wave-vector in y direction
            kz_relative : float,
                Non-dimensional wave-vector in z direction

            Returns
            -------
            q3_array : float, dimension (N-q, N-q, 3)
                3d q_array of wavevector in all direction on a 2D dectector grid
    '''
    ky, kz = np.meshgrid(ky_relative, kz_relative)
    kx = np.sqrt(1.0 - np.square(ky) - np.square(kz))
    qx, qy, qz = kx - 1.0, ky, kz
    q3_array = np.stack((qx, qy, qz), axis=-1) * k0
    return q3_array

def get_r_array(pos, h, rc, nnlist=None, atoms_add = None):
    '''Get array of pairwise distances

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            nnlist : List,
                List of neighbors -- similar nindex
            atoms_add : int,
                Number of atoms to consider for getting r_array

            Returns
            -------
            r_array : float, dimension (arbitrary,)
                array of all pairwise distances for the specified cutoff
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    if atoms_add != None:
        nparticles = atoms_add
    else:
        nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    r_list = []
    for i in range(nparticles):
        ri = np.array([pos[i,:]])
        ind = index[i]
        rj = pos[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dis = np.linalg.norm(drij, axis=1)
        indo=np.where(dis>rc)
        drij=np.delete(drij, indo, axis=0)
        r_list.extend(drij.tolist())

    r_array = np.array(r_list)
    return r_array

def get_r_array_ref(pos, pos_ref, h, rc, nnlist=None):
    '''Get array of pairwise distances

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            pos_ref : float, dimension (natoms, 3)
                Position of all atoms in first/reference configuration
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            r_array : float, dimension (arbitrary,)
                array of all pairwise distances for the specified cutoff
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)
    
    if pos.shape[0] != pos_ref.shape[0]:
        raise TypeError('Initial configuration and the desired configuration do not have the same number of atoms/particles')
    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    r_list = []
    for i in range(nparticles):
        ri = np.array([pos[i,:]])
        ind = index[i]
        rj = pos_ref[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        r_list.extend(drij.tolist())

    r_array = np.array(r_list)
    return r_array

def s_q3_from_pos(q3_flat, pos, nparticles, intensity = False, smear=False,ddq=0,r_sq=None):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''
    s_flat = np.zeros(q3_flat.shape[0], dtype=np.complex64)
    for i in range(q3_flat.shape[0]): # note: range -> prange
        q3 = q3_flat[i,:]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        adj_ISF = np.exp(-1j*np.dot(pos,q3.T)).sum()
        s_flat[i]=adj_ISF*np.conj(adj_ISF)/nparticles
    return s_flat.real

def s_q3_from_pos_par(q3_flat, r_array, rc, rho, nparticles, intensity = False, smear=False,ddq=0,r_sq=None):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            rho : float,
                Atomic density
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            smear :  boolean
                If smearing of density field is being considered
            ddq : float,
                Width of the gaussian smear being used
            r_sq : float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''
    s_flat = np.zeros(q3_flat.shape[0], dtype=np.complex64)
    for i in range(q3_flat.shape[0]): # note: range -> prange
        q3 = q3_flat[i,:]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        if smear:
            exp_miqr = np.exp(-1j * np.dot(r_array,q3.T)).flatten()*np.exp(-(0.5*ddq**2)*r_sq**2)
            s_flat[i] = (1.0 + np.sum(exp_miqr) / nparticles)
        else:
            exp_miqr = np.exp(-1j * np.dot(r_array,q3.T))
            int_0_rc = (rc*q*np.cos(rc*q) - np.sin(rc*q))/np.power(q,3)  # integral from 0 to rc
            s_flat[i] = (1.0 + 4.0*np.pi*rho*int_0_rc + np.sum(exp_miqr) / nparticles)
    return s_flat.real

def s_q_position_par(q3_pos):
    '''Get s(q) from positions on the q3_pos vector

            Parameters
            ----------
            q3_pos: float, dimension(natoms+1, 3)
                array of first row as wave-vector and the remaining rows as position of atoms
                -- function created for use in conjunction with multiprocessing

            Returns
            -------
            s_positon : float
                s(q) over the q3_pos wave-vector
    '''
    q3_array = q3_pos[0:1,:]
    pos = q3_pos[1:,:]
    N_atoms = q3_pos.shape[0]-1
    s_position= np.zeros(1,dtype = np.complex)
    q3 = q3_array[0:1,:]
    q = np.linalg.norm(q3)
    adj_ISF = np.exp(-1j*np.dot(pos,q3.T)).sum()
    s_position=adj_ISF*np.conj(adj_ISF)/N_atoms
    return s_position

def I_q3_from_pos_par(q3_flat, r_array, rc, rho, nparticles, ff, smear = False, ddq = 0, r_sq = None):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            rho : float,
                Atomic density
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            smear :  boolean
                If smearing of density field is being considered
            ddq : float,
                Width of the gaussian smear being used
            r_sq : float, dimension (N_pairs,)
                Squared distances from r_array
            ff : float,
                form factor over k-space for a given scatterer

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''
    s_flat = np.zeros(q3_flat.shape[0])
    for i in range(q3_flat.shape[0]): # note: range -> prange
        q3 = q3_flat[i,:]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        if smear:
            exp_miqr = (((2*np.pi)**0.5*ddq)**3)*np.exp(-1j * np.dot(r_array,q3.T)).flatten()*np.exp(-(0.5*ddq**2)*r_sq**2)
            int_0_rc = 0*(-rc*((2*np.pi)**0.5)*np.cos(rc*q)*np.exp(-0.5*(ddq**2)*rc**2)/ddq)  # integral from 0 to rc
            s_flat[i] = ((ddq*(2*np.pi)**0.5)**3 + 4.0*np.pi*rho*int_0_rc + np.sum(exp_miqr) / nparticles)*ff[i]**2
        else:
            exp_miqr = np.exp(-1j * np.dot(r_array,q3.T))
            int_0_rc = (rc*q*np.cos(rc*q) - np.sin(rc*q))/np.power(q,3)  # integral from 0 to rc
            s_flat[i] = (1.0 + 4.0*np.pi*rho*int_0_rc + np.sum(exp_miqr) / nparticles)*ff[i]**2
    return s_flat.real

def ISF_from_pos_par(posit, s_ref=np.zeros((10,10,10)), N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0,2]), dump = False, grid = False):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            grid : boolean
                Return whole minigrid instead of just indices

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h=posit[:2,:].T
    pos=posit[2:,:]
    '''
    s_flat = np.zeros(q3_flat.shape[0], dtype=np.complex64)
    for i in prange(q3_flat.shape[0]): # note: range -> prange
        q3 = q3_flat[i,:]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        exp_miqr = np.exp(-1j * np.dot(r_array, q3))
        int_0_rc = (rc*q*np.cos(rc*q) - np.sin(rc*q))/np.power(q,3)  # integral from 0 to rc
        s_flat[i] = np.sum(exp_miqr) / nparticles # +4.0*np.pi*rho*int_0_rc 
    '''
    if dump:
        x=np.linspace(0,-h[0,0]+h[0,1],N+1)
        y=np.linspace(0,-h[1,0]+h[1,1],N+1)
        z=np.linspace(0,-h[2,0]+h[2,1],N+1)
    else:
        x=np.linspace(h[0,0],h[0,1],N+1)
        y=np.linspace(h[1,0],h[1,1],N+1)
        z=np.linspace(h[2,0],h[2,1],N+1)
    boxl=np.mean(h[:,1]-h[:,0])
    hx=h[0,1]-h[0,0]
    hy=h[1,1]-h[1,0]
    hz=h[2,1]-h[2,0]
    [X,Y,Z]=np.meshgrid(x,y,z)
    n_of_r=np.zeros(X.shape)
    NP=pos.shape[0]
    box_diag = h[:,1]-h[:,0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i,:])/box_diag
        else:
            s_pos = (pos[i,:]-h[:,0])/box_diag
        kx=int(np.floor(s_pos[0]*N))
        ky=int(np.floor(s_pos[1]*N))
        kz=int(np.floor(s_pos[2]*N))
        if (kx>=cs) and (kx<N-cs):
            indx=np.linspace(kx-cs,kx+cs,2*cs+1,dtype=int)
        elif kx<cs:
            indx=np.append(np.linspace(N-(cs-kx),N-1,cs-kx,dtype=int),np.linspace(0,kx+cs,2*cs+1-(cs-kx),dtype=int))
        elif kx>=N-cs:
            indx=np.append(np.linspace(kx-cs,N-1,cs+N-kx,dtype=int),np.linspace(0,cs+kx-N,cs+kx-N+1,dtype=int))
        if (ky>=cs) and (ky<N-cs):
            indy=np.linspace(ky-cs,ky+cs,2*cs+1,dtype=int)
        elif ky<cs:
            indy=np.append(np.linspace(N-(cs-ky),N-1,cs-ky,dtype=int),np.linspace(0,ky+cs,2*cs+1-(cs-ky),dtype=int))
        elif ky>=N-cs:
            indy=np.append(np.linspace(ky-cs,N-1,cs+N-ky,dtype=int),np.linspace(0,cs+ky-N,cs+ky-N+1,dtype=int))
        if (kz>=cs) and (kz<N-cs):
            indz=np.linspace(kz-cs,kz+cs,2*cs+1,dtype=int)
        elif kz<cs:
            indz=np.append(np.linspace(N-(cs-kz),N-1,cs-kz,dtype=int),np.linspace(0,kz+cs,2*cs+1-(cs-kz),dtype=int))
        elif kz>=N-cs:
            indz=np.append(np.linspace(kz-cs,N-1,cs+N-kz,dtype=int),np.linspace(0,cs+kz-N,cs+kz-N+1,dtype=int))
        indexes=np.ix_(indy,indx,indz)
        [XX,YY,ZZ]=np.meshgrid(x[indx],y[indy],z[indz])
        n_of_r[indexes]+=norm3d(pos[i,:],XX,YY,ZZ,boxl/wdt,hx,hy,hz)

    s_r=np.fft.fftshift(np.fft.fftn(n_of_r))
    if grid:
        return s_r
    else:
        s_r=s_ref*np.flip(np.flip(np.flip(s_r,axis=0),axis=1),axis=2)
        xxx=np.arange(N//2-ms-1,N//2+ms)
        NN=len(xxx)
        index2=np.ix_(xxx,xxx,xxx)
        s_val=np.zeros((NN)**3)
        s_val[:]=np.reshape(s_r[index2],(1,-1))

        return s_val[ind_need]

def get_h_from_lines(lines,start,h_full=True):
    '''Get box dimensions from lines read from dumpfile

                Parameters
                ----------
                lines : List
                    List of all line
                h_full : boolean
                    True if all box lengths are required as a vector; False if hi and lo values of box sides are needed
                start : int
                    Line number to start from

                Returns
                -------
                h : float, dimension (1, 3) if True else (3,2)
                    Periodic box size h = (c1|c2|c3)

    '''
    if h_full:
        h=np.zeros([1,3])
        for i in range(3):
            a=np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start+i].strip()),dtype=float)
            h[0,i]=a[1]-a[0]
    else:    
        h=np.zeros([3,2])
        for i in range(3):
            a=np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start+i].strip()),dtype=float)
            h[i,:]=np.array([a[0],a[1]])
    return h

def get_pos_from_lines(lines,start,atoms,add='position'):
    '''Get box dimensions from lines read from dumpfile

                    Parameters
                    ----------
                    lines : List
                        List of all line
                    atoms : int
                        Number of atoms per frame
                    start : int
                        Line number to start from
                    add: str,
                        If dumpfile has just positions or positions and velocities

                    Returns
                    -------
                    pos : float, dimension (atoms, 3)
                        Position of all atoms

    '''
    if add=='position':
        raw=np.zeros([atoms,5])
    else:
        raw=np.zeros([atoms,8])
    for i in range(atoms):
        raw[i,:]=np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start+i].strip()),dtype=float)
        
    return raw

def f_weight_array(g,r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array
            g: float, dimension (N_pairs,)
                g(r) over all the r_array distances

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    weight=np.zeros(np.size(r))
    weight[r<ri]=1.0+(g[r<ri]-1.0)
    weight[(r>=ri) & (r<=rc)]=1.0+0.5*(1.0+np.cos(np.pi*((r[(r>=ri) & (r<=rc)]-ri)/(rc-ri))))*(g[(r>=ri) & (r<=rc)]-1.0)
    return weight

def f_weight_vec(r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    weight=np.zeros(np.size(r))
    weight[r<ri]=(r[r<ri])
    weight[(r>=ri) & (r<=rc)]=0.5*(1.0+np.cos(np.pi*((r[(r>=ri) & (r<=rc)]-ri)/(rc-ri))))*(r[(r>=ri) & (r<=rc)])
    return weight

def f_weight(r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    if r<ri:
        weight=1
    elif r>rc:
        weight=0
    else:
        weight=0.5*(1.0+cos(np.pi*((r-ri)/(rc-ri))))
    return weight

def auto_corr(s_cor,frames,g_2=True):
    '''Auto-correlation computation

            Parameters
            ----------
            s_cor : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            g_2 : boolean
                True if g_2(q) normalization is required; False if F(q,t) normalization is required

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of correlation values
    '''
    ts_cor=np.zeros(frames)
    for i in range(frames):
        if i==0:
            f=s_cor
            g=s_cor
        else:
            f=s_cor[i:]
            g=s_cor[:-i]
        ts_cor[i]=np.mean(f*g)
    if g_2:	
        return ts_cor/s_cor.mean()**2
    else:
        return ts_cor/ts_cor[0]

def ISF_corr(s_cor,frames,normed=True):
    '''Auto-correlation computation

            Parameters
            ----------
            s_cor : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            normed : boolean
                Requirement of normalization (used mainly for F(q,t)

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of correlation values
    '''
    ts_cor=np.zeros(frames)
    for i in range(frames):
        if i==0:
            f=s_cor
            g=s_cor
        else:
            f=s_cor[i:]
            g=s_cor[:-i]
        ts_cor[i]=np.mean(f*np.conj(g))
    if normed:
        return ts_cor/ts_cor[0]
    else:
        return ts_cor

def pbc1d(r,h):
    '''calculate distance vector between i and j

            Considering periodic boundary conditions (PBC)

            Parameters
            ----------
            r : float, dimension (1, 3)
                distance vectors of atom pairs (Angstrom)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)

            Returns
            -------
            r_ : float, dimension (npairs, 3)
                modified distance vectors of atom pairs considering PBC (Angstrom)

    '''
    s=r/h
    s=s-np.round(s)
    return s*h

def norm3d(arr,x,y,z,sig,hx,hy,hz):
    '''Density field generation

            Parameters
            ----------
            arr : float, dimension (3,)
                Array of mean width density field in the x, y , z directions respectively
            x: float, dimension (Nx, Nx, Nx)
                mesh grid of x coordinates
            y: float, dimension (Nx, Nx, Nx)
                mesh grid of y coordinates
            z: float, dimension (Nx, Nx, Nx)
                mesh grid of z coordinates
            sig: float
                Smear width of density gaussian
            hx: float
                box length in x direction
            hy: float
                box length in y direction
            hz: float
                box length in z direction

            Returns
            -------
            gaussian density : float, dimension (Nx, Nx, Nx)
                Gaussian density on meshgrid defined by x, y, z
    '''
    mux,muy,muz=arr[0],arr[1],arr[2]
    return (1/(np.sqrt(2*np.pi*sig**2)**3))*(np.exp(-0.5*(pbc1d(x-mux,hx)/sig)**2)*np.exp(-0.5*(pbc1d(y-muy,hy)/sig)**2)*np.exp(-0.5*(pbc1d(z-muz,hz)/sig)**2))

def convert_density(X,Y,Z,NP,pos,boxl,wdt,hx,hy,hz):
    '''Convert to atomic density using a gaussian smear

            Parameters
            ----------
            arr : float, dimension (3,)
                Array of mean width density field in the x, y , z directions respectively
            X: float, dimension (Nx, Nx, Nx)
                mesh grid of x coordinates
            Y: float, dimension (Nx, Nx, Nx)
                mesh grid of y coordinates
            Z: float, dimension (Nx, Nx, Nx)
                mesh grid of z coordinates
            sig: float
                Smear width of density gaussian
            boxl: float
                box length in x direction
            hx: float
                box length in x direction
            hy: float
                box length in y direction
            hz: float
                box length in z direction
            NP : int,
                Total number of atoms
            pos : float, dimension (NP, 3)
                Position of all atoms

            Returns
            -------
            gaussian density : float, dimension (Nx, Nx, Nx)
                Gaussian density on meshgrid defined by x, y, z
    '''
    n_of_r=np.zeros(X.shape)
    sig=boxl/wdt
    a=0
    for i in range(NP):
    #	n_of_r+=norm3d(X,Y,Z,pos[i,0],pos[i,1],pos[i,2],boxl/wdt,hx,hy,hz)
        pbcx=((pos[i,0]-X)/hx-np.round_((pos[i,0]-X)/hx,0,a))*hx
        pbcy=((pos[i,1]-Y)/hy-np.round_((pos[i,1]-Y)/hy,0,a))*hy
        pbcz=((pos[i,2]-Z)/hz-np.round_((pos[i,2]-Z)/hz,0,a))*hz
        n_of_r+=(1/(np.sqrt(2*np.pi*sig**2)**3))*(np.exp(-0.5*(pbcx/sig)**2)*np.exp(-0.5*(pbcy/sig)**2)*np.exp(-0.5*(pbcz/sig)**2))

    return n_of_r

def s_q_from_pos_smear_analytical(q_array, pos, h, rc, rho, nnlist=None):
    '''Obtain s(q) from analytical expression -- internally computes r_array from g(r)

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            rho : float,
                Atomic density
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i,:]])
        ind = index[i]
        rj = pos[ind,:]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    r_array = np.array(dist)
#    r_array= f_weight_vec(r_array,rc)
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        sinqr_div_r = np.exp(-1j * r_array* q)
        int_0_rc = (rc*q*np.cos(rc*q) - np.sin(rc*q))/np.power(q,3)  # integral from 0 to rc
        s_array[i] = 1.0 + 4.0*np.pi*rho*int_0_rc +  np.sum(sinqr_div_r)/(nparticles)#np.sum(sinqr_div_r)/(nparticles)
    return s_array.real

def accum_np(accmap, a, func=np.mean):
    indices = np.where(np.ediff1d(accmap, to_begin=[1],
                                  to_end=[1]))[0]
    vals = np.zeros(len(indices) - 1)
    for i in range(len(indices) - 1):
        vals[i] = func(a[indices[i]:indices[i+1]])
    return vals

def s_q_from_pos_smear(posit, N=200, wdt=500, cs=3, ms=30, dump = False, intensity = False, movie_plot = False,ISF=False, structure_factor = False, correction = False, correction_grid = None, q_magnitude = None, ind_need = None):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            intensity : boolean
                If I(q) needs to be computed
            ISF : boolean
                If F(q) needs to be computed
            structure : boolean
                If s(q) needs to be computed
            correction_grid : float, dimension (N,N,N)
                Correction grid value
            correction : boolean
                If correction factor is needed
            q_magnitude : float, dimension (N,N,N)
                Needeed to compute I(q) from s(q)
            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    if correction is False:
        if (int(intensity)+int(ISF)+int(structure_factor)) != 1:
            raise TypeError('Choose one and only one output entity')
    h=posit[:2,:].T
    pos=posit[2:,:]
    boxl=np.mean(h[:,1]-h[:,0])
    hx=h[0,1]-h[0,0]
    hy=h[1,1]-h[1,0]
    hz=h[2,1]-h[2,0]
    s =pos/boxl
    s=s-0.5-np.round(s-0.5)
    pos = boxl*(s+0.5)
    delx=hx/N
    if dump:
        x=np.linspace(0,hx-delx,N)
        y=np.linspace(0,hy-delx,N)
        z=np.linspace(0,hz-delx,N)
    else:
        x=np.linspace(h[0,0],h[0,1]-delx,N)
        y=np.linspace(h[1,0],h[1,1]-delx,N)
        z=np.linspace(h[2,0],h[2,1]-delx,N)
    [X,Y,Z]=np.meshgrid(x,y,z)
    n_of_r=np.zeros(X.shape)
    s_r=np.zeros(X.shape,dtype=np.complex)
    NP=pos.shape[0]
    box_diag = h[:,1]-h[:,0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i,:])/box_diag
        else:
            s_pos = (pos[i,:]-h[:,0])/box_diag
        kx=int(np.floor(s_pos[0]*N))
        ky=int(np.floor(s_pos[1]*N))
        kz=int(np.floor(s_pos[2]*N))
        if (kx>=cs) and (kx<N-cs):
            indx=np.linspace(kx-cs,kx+cs,2*cs+1,dtype=int)
        elif kx<cs:
            indx=np.append(np.linspace(N-(cs-kx),N-1,cs-kx,dtype=int),np.linspace(0,kx+cs,2*cs+1-(cs-kx),dtype=int))
        elif kx>=N-cs:
            indx=np.append(np.linspace(kx-cs,N-1,cs+N-kx,dtype=int),np.linspace(0,cs+kx-N,cs+kx-N+1,dtype=int))
        if (ky>=cs) and (ky<N-cs):
            indy=np.linspace(ky-cs,ky+cs,2*cs+1,dtype=int)
        elif ky<cs:
            indy=np.append(np.linspace(N-(cs-ky),N-1,cs-ky,dtype=int),np.linspace(0,ky+cs,2*cs+1-(cs-ky),dtype=int))
        elif ky>=N-cs:
            indy=np.append(np.linspace(ky-cs,N-1,cs+N-ky,dtype=int),np.linspace(0,cs+ky-N,cs+ky-N+1,dtype=int))
        if (kz>=cs) and (kz<N-cs):
            indz=np.linspace(kz-cs,kz+cs,2*cs+1,dtype=int)
        elif kz<cs:
            indz=np.append(np.linspace(N-(cs-kz),N-1,cs-kz,dtype=int),np.linspace(0,kz+cs,2*cs+1-(cs-kz),dtype=int))
        elif kz>=N-cs:
            indz=np.append(np.linspace(kz-cs,N-1,cs+N-kz,dtype=int),np.linspace(0,cs+kz-N,cs+kz-N+1,dtype=int))
        indexes=np.ix_(indy,indx,indz)
        [XX,YY,ZZ]=np.meshgrid(x[indx],y[indy],z[indz])
        n_of_r[indexes]+=norm3d(pos[i,:],XX,YY,ZZ,boxl/wdt,hx,hy,hz)
    s_r=np.fft.fftshift(np.fft.fftn(n_of_r))*(boxl/(N))**3
    p_r = s_r/NP
    s_r=s_r*np.conj(s_r)/NP
    if correction:
        if ISF:
            return (p_r/NP)**-1
        else:
            return s_r**-1
    else:
        xxx=np.linspace(N//2-ms,N//2+ms,2*ms+1,dtype=np.int)
        NN=len(xxx)
        index2=np.ix_(xxx,xxx,xxx)
        [XXX,YYY,ZZZ]=np.meshgrid(xxx,xxx,xxx)
        if correction_grid is None:
            #s_r = correction_grid*s_r
            #p_r = correction_grid*p_r
            if ind_need is None:
                if intensity:
                    if q_magnitude is None:
                        raise TypeError("magnitude of q-space is necessary for computing the intensity")
                    ret_val=s_r*form_factor_analytical(q_magnitude)**2
                    return ret_val
                elif ISF is True:
                    return p_r
                elif structure_factor is True:
                    return s_r
            else:
                if intensity:
                    if q_magnitude is None:
                        raise TypeError("magnitude of q-space is necessary for computing the intensity")
                    ret_val=s_r*form_factor_analytical(q_magnitude)**2
                    I_q =ret_val[index2].reshape(-1,1)
                    return I_q[ind_need].flatten()
                elif ISF is True:
                    p_r = p_r[index2].reshape(-1,1)
                    return p_r[ind_need].flatten()
                elif structure_factor is True:
                    s_r = s_r[index2].reshape(-1,1)
                    return s_r[ind_need].flatten()

def s_q_from_pos_smear_array(pos, h, N=200, wdt=500, cs=3, ms=30, uniform_density = True, dump = False, correction_grid = None):
    '''Get I(q) from positions on a line of spherically averaged wave-vectors

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            dump : boolean
                If dumpfile is being read
            correction_grid : float, dimension (N,N,N)
                Correction grid value
            Returns
            -------
            r_un : float, dimension (arb,)
                q_value array
            signa : float, dimension (arb,)
                angle averaged s(q) on the r_un line of wavevectors
    '''
    h=pos[:2,:].T
    boxl=np.mean(h[:,1]-h[:,0])
    s_r = s_q_from_pos_smear(pos,N = N, wdt = wdt, cs = cs, ms = ms, dump=dump, structure_factor = True)
    s_r = correction_grid*s_r
    xxx=np.linspace(N//2-ms,N//2+ms,2*ms+1,dtype=np.int)
    NN=len(xxx)
    index2=np.ix_(xxx,xxx,xxx)
    [XXX,YYY,ZZZ]=np.meshgrid(xxx,xxx,xxx)
    s_val=np.zeros([(NN)**3,2])
    posit=np.zeros([(NN)**3,3])
    [XXX,YYY,ZZZ]=np.meshgrid(xxx,xxx,xxx)
    posit[:,0]=np.reshape(XXX,(1,-1))
    posit[:,1]=np.reshape(YYY,(1,-1))
    posit[:,2]=np.reshape(ZZZ,(1,-1))
    s_val[:,0]=np.linalg.norm((posit-N//2)/boxl,axis=1)
    s_val[:,1]=np.reshape(s_r[index2],(1,-1))
    s_val=s_val[np.argsort(s_val[:,0])]
    r_un, ia, idx=np.unique(s_val[:,0],return_index=True, return_inverse=True)
    signa=accum_np(idx,s_val[:,1])
    return r_un, signa

def s_q_from_pos_smear_par(posit, N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0,2]), dump = False):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h=posit[:2,:].T
    pos=posit[2:,:]
    if dump:
        x=np.linspace(0,hx-delx,N)
        y=np.linspace(0,hy-delx,N)
        z=np.linspace(0,hz-delx,N)
    else:
        x=np.linspace(h[0,0],h[0,1]-delx,N)
        y=np.linspace(h[1,0],h[1,1]-delx,N)
        z=np.linspace(h[2,0],h[2,1]-delx,N)
    boxl=np.mean(h[:,1]-h[:,0])
    hx=h[0,1]-h[0,0]
    hy=h[1,1]-h[1,0]
    hz=h[2,1]-h[2,0]
    s =pos/boxl
    s=s-0.5-np.round(s-0.5)
    pos = boxl*(s+0.5)
    [X,Y,Z]=np.meshgrid(x,y,z)
    n_of_r=np.zeros(X.shape)
    NP=pos.shape[0]
    box_diag = h[:,1]-h[:,0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i,:])/box_diag
        else:
            s_pos = (pos[i,:]-h[:,0])/box_diag
        kx=int(np.floor(s_pos[0]*N))
        ky=int(np.floor(s_pos[1]*N))
        kz=int(np.floor(s_pos[2]*N))
        if (kx>=cs) and (kx<N-cs):
            indx=np.linspace(kx-cs,kx+cs,2*cs+1,dtype=int)
        elif kx<cs:
            indx=np.append(np.linspace(N-(cs-kx),N-1,cs-kx,dtype=int),np.linspace(0,kx+cs,2*cs+1-(cs-kx),dtype=int))
        elif kx>=N-cs:
            indx=np.append(np.linspace(kx-cs,N-1,cs+N-kx,dtype=int),np.linspace(0,cs+kx-N,cs+kx-N+1,dtype=int))
        if (ky>=cs) and (ky<N-cs):
            indy=np.linspace(ky-cs,ky+cs,2*cs+1,dtype=int)
        elif ky<cs:
            indy=np.append(np.linspace(N-(cs-ky),N-1,cs-ky,dtype=int),np.linspace(0,ky+cs,2*cs+1-(cs-ky),dtype=int))
        elif ky>=N-cs:
            indy=np.append(np.linspace(ky-cs,N-1,cs+N-ky,dtype=int),np.linspace(0,cs+ky-N,cs+ky-N+1,dtype=int))
        if (kz>=cs) and (kz<N-cs):
            indz=np.linspace(kz-cs,kz+cs,2*cs+1,dtype=int)
        elif kz<cs:
            indz=np.append(np.linspace(N-(cs-kz),N-1,cs-kz,dtype=int),np.linspace(0,kz+cs,2*cs+1-(cs-kz),dtype=int))
        elif kz>=N-cs:
            indz=np.append(np.linspace(kz-cs,N-1,cs+N-kz,dtype=int),np.linspace(0,cs+kz-N,cs+kz-N+1,dtype=int))
        indexes=np.ix_(indy,indx,indz)
        [XX,YY,ZZ]=np.meshgrid(x[indx],y[indy],z[indz])
        n_of_r[indexes]+=norm3d(pos[i,:],XX,YY,ZZ,boxl/wdt,hx,hy,hz)

    s_r=np.fft.fftshift(np.fft.fftn(n_of_r))
    s_r=s_r*np.conj(s_r)
    xxx=np.arange(N//2-ms-1,N//2+ms)
    NN=len(xxx)
    index2=np.ix_(xxx,xxx,xxx)
    s_val=np.zeros((NN)**3)
    s_val[:]=np.reshape(s_r[index2],(1,-1))
    return s_val[ind_need]

def I_q_from_pos_smear_par(posit, N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0,2]), dump = False, movie_plot = False, ISF = False, coarse_grain =False, grid_type = '3D'):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            ISF : boolean
                If F(q) needs to be computed or s(q)
            movie_plot : boolean
                If a 2D frame needs to be generated for movie of 2D snapshots
            coarse_grain : int
                number of grid points to combine to coarse grain the signal
            grid_type : str
                '3D' or 2D'

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h=posit[:2,:].T
    pos=posit[2:,:]
    boxl=np.mean(h[:,1]-h[:,0])
    hx=h[0,1]-h[0,0]
    hy=h[1,1]-h[1,0]
    hz=h[2,1]-h[2,0]
    s =pos/boxl
    s=s-0.5-np.round(s-0.5)
    pos = boxl*(s+0.5)
    delx=hx/N
    if dump:
        x=np.linspace(0,hx-delx,N)
        y=np.linspace(0,hy-delx,N)
        z=np.linspace(0,hz-delx,N)
    else:
        x=np.linspace(h[0,0],h[0,1]-delx,N)
        y=np.linspace(h[1,0],h[1,1]-delx,N)
        z=np.linspace(h[2,0],h[2,1]-delx,N)
    [X,Y,Z]=np.meshgrid(x,y,z)
    n_of_r=np.zeros(X.shape)
    s_r=np.zeros(X.shape,dtype=np.complex)
    NP=pos.shape[0]
    box_diag = h[:,1]-h[:,0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i,:])/box_diag
        else:
            s_pos = (pos[i,:]-h[:,0])/box_diag
        kx=int(np.floor(s_pos[0]*N))
        ky=int(np.floor(s_pos[1]*N))
        kz=int(np.floor(s_pos[2]*N))
        if (kx>=cs) and (kx<N-cs):
            indx=np.linspace(kx-cs,kx+cs,2*cs+1,dtype=int)
        elif kx<cs:
            indx=np.append(np.linspace(N-(cs-kx),N-1,cs-kx,dtype=int),np.linspace(0,kx+cs,2*cs+1-(cs-kx),dtype=int))
        elif kx>=N-cs:
            indx=np.append(np.linspace(kx-cs,N-1,cs+N-kx,dtype=int),np.linspace(0,cs+kx-N,cs+kx-N+1,dtype=int))
        if (ky>=cs) and (ky<N-cs):
            indy=np.linspace(ky-cs,ky+cs,2*cs+1,dtype=int)
        elif ky<cs:
            indy=np.append(np.linspace(N-(cs-ky),N-1,cs-ky,dtype=int),np.linspace(0,ky+cs,2*cs+1-(cs-ky),dtype=int))
        elif ky>=N-cs:
            indy=np.append(np.linspace(ky-cs,N-1,cs+N-ky,dtype=int),np.linspace(0,cs+ky-N,cs+ky-N+1,dtype=int))
        if (kz>=cs) and (kz<N-cs):
            indz=np.linspace(kz-cs,kz+cs,2*cs+1,dtype=int)
        elif kz<cs:
            indz=np.append(np.linspace(N-(cs-kz),N-1,cs-kz,dtype=int),np.linspace(0,kz+cs,2*cs+1-(cs-kz),dtype=int))
        elif kz>=N-cs:
            indz=np.append(np.linspace(kz-cs,N-1,cs+N-kz,dtype=int),np.linspace(0,cs+kz-N,cs+kz-N+1,dtype=int))
        indexes=np.ix_(indy,indx,indz)
        [XX,YY,ZZ]=np.meshgrid(x[indx],y[indy],z[indz])
        n_of_r[indexes]+=norm3d(pos[i,:],XX,YY,ZZ,boxl/wdt,hx,hy,hz)

    s_r=np.fft.fftshift(np.fft.fftn(n_of_r))*(boxl/(N+1))**3
    if ISF:
        p_temp = s_r
    s_r=s_r*np.conj(s_r)/NP
    if coarse_grain:
        s_r_old = s_r.copy()
        N=N//2
        s_r = np.zeros((N,N,N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    s_r[i,j,k]=np.sum(np.sum(np.sum(s_r_old[2*i:2*i+2,2*j:2*j+2,2*k:2*k+2],axis=0),axis=1))
    xxx=np.linspace(N//2-ms,N//2+ms,2*ms+1,dtype=np.int)
    NN=len(xxx)
    index2=np.ix_(xxx,xxx,xxx)
    [XXX,YYY,ZZZ]=np.meshgrid(xxx,xxx,xxx)
    if grid_type=='3D':
        posit=np.zeros([(NN)**3,3])
        posit[:,0]=np.reshape(XXX,(1,-1))
        posit[:,1]=np.reshape(YYY,(1,-1))
        posit[:,2]=np.reshape(ZZZ,(1,-1))
        s_val=np.zeros((NN)**3)
        s_val[:]=np.reshape(s_r[index2],(1,-1))
    else:
        posit=np.zeros([(NN)**2,3])
        [XXX,YYY,ZZZ]=np.meshgrid(xxx,xxx,xxx)
        posit[:,0]=np.reshape(XXX[:,NN//2,:],(1,-1))
        posit[:,1]=np.reshape(YYY[:,NN//2,:],(1,-1))
        posit[:,2]=np.reshape(ZZZ[:,NN//2,:],(1,-1))
        ts_val=s_r[index2]
        s_val=np.zeros((NN)**2)
        s_val[:]=np.reshape(ts_val[:,NN//2,:],(1,-1))
    if coarse_grain:
        d_val=np.linalg.norm(2.0*(posit-N//2)/boxl,axis=1)
    else:
        d_val=np.linalg.norm((posit-N//2)/boxl,axis=1)
    #ret_val=np.exp((2*np.pi*d_val)**2*(boxl/wdt)**2)*s_val*form_factor_analytical(2*np.pi*d_val)**2
    s_val=np.exp((2*np.pi*d_val)**2*(boxl/wdt)**2)*s_val
    ret_val=s_val*form_factor_analytical(2*np.pi*d_val)**2
    #ret_val = ret_val.reshape(XXX.shape)
    s_val = s_val.reshape(-1,1)
    if ISF:
        if grid_type =='3D':
            p_val=np.zeros((NN)**3,dtype=np.complex)
            p_val[:]=np.reshape(p_temp[index2],(1,-1))
        else:
            ps = p_temp[index2]
            p_val=np.zeros((NN)**2,dtype=np.complex)
            p_val[:]=np.reshape(ps[:,NN//2,:],(1,-1))
        p_r=np.exp((2*np.pi*d_val)**2*(boxl/wdt)**2)*p_val#*form_factor_analytical(2*np.pi*d_val)
        p_r=p_r*np.conj(p_r)/NP
        if movie_plot:
            ret_grid = ret_val.reshape((NN,NN,NN))
            return ret_grid[NN//2,:,:]
        else:
            return p_r[ind_need]
    else:
        if movie_plot:
            ret_grid = ret_val.reshape((NN,NN,NN))
            return ret_grid[NN//2,:,:]
        else:
            return s_val[ind_need].flatten()

def form_factor_analytical(q_val, atom_type="Ar"):
    '''Computing form factor at a given wave-vector

            Parameters
            ----------
            q_val : float, dimension (N-q, 3)
                Array of all wave-vectors for form factor computation
            atom_type : str
                Atom type for form factor estimation

            Returns
            -------
            f_val : float, dimension (q_val,)
                form factor computed for given q_val array
    '''
    # Default values for Ar, need to be changed for DPN simulation
    if atom_type == "Ar":
        a = np.array([7.4845,6.7723,0.6539,1.6442])
        b = np.array([0.9072,14.8407,43.8983,33.3929])
        c = 1.4445
    elif atom_type == "C":
        a = np.array([2.31,1.02,1.5886,0.865])
        b = np.array([20.8439,10.2075,0.5687,51.6512])
        c = 0.2156
    elif atom_type == "H":
        a = np.array([0.489918,0.262003,0.196767,0.049879])
        b = np.array([20.6593,7.74039,49.5519,2.20159])
        c = 0.001305
    else:
        a = np.array([0,0,0,0])
        b = np.array([0,0,0,0])
        c = 0
    f_val = c + np.sum(a*np.exp(-b*(q_val/(4*np.pi))**2))

    return f_val

def optical_contrast(s):
    '''Compute optical contrast from an s(q) or I(q) array

            Parameters
            ----------
            s : float, dimension (N,)
                Array of s(q) or I(q) in either q-space or t-space

            Returns
            -------
            contrast : float,
                optical contrast
    '''
    return ((s**2).mean()-(s.mean())**2)/(s.mean())**2

def cross_corr(s,frames):
    '''Time cross-correlation (two-time correlation) computation

            Parameters
            ----------
            s : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of time cross-correlation grid
    '''
    ts = np.zeros((frames,frames))
    for i in range(frames):
        for j in range(i,frames):
            if i==j:
                f = s[i:]
                g = s[i:]
            else:
                f = s[j:]
                g = s[i:-(j-i)]
            ts[i,j] = (np.mean(f*g)-np.mean(f)*np.mean(g))/(((np.mean(f**2)-np.mean(f)**2)**0.5)*((np.abs(np.mean(g**2)-np.mean(g)**2)**0.5)))
            if i!=j:
                ts[j,i] = ts[i,j]
    return ts

def config_repeater(pos_a,h,size = 2):
    '''Repeat the configuration in case g(r) cutoff is too big for the current box

            Parameters
            ----------
            pos_a : float, dimension (npairs, 3)
                distance vectors of atom pairs (Angstrom)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            size : int
                Number of times to repeat the vox in each direction

            Returns
            -------
            new_pos : float, dimension (npairs*size**3, 3)
                modified positions of atom pairs after box repetition
            new_h : float, dimension (3, 3)
                modified Periodic box size h = (c1|c2|c3)

    '''
    pos = pos_a.copy()
    boxl = np.mean(np.diag(h))
    new_h = h*size
    new_pos = np.zeros((pos.shape[0]*size**3,pos.shape[1]))
    atoms = pos.shape[0]
    ct=0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                new_pos[ct*atoms:(ct+1)*atoms,:]=pos[:,:]+boxl*np.repeat(np.array([[i,j,k]]),atoms,axis=0)
                ct+=1
    return new_pos, new_h

class bcolors:
    RED = '\033[31m'
    GRN = '\033[32m'
    YEL = '\033[33m'
    BLU = '\033[34m'
    MAG = '\033[35m'
    CYN = '\033[36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    UNDERLINE = '\033[4m'

class test_util():
    def __init__(self,box_length = 17.44024100780622, rc = 4.0, filename = 'dump.ljtest25k_1000',smear = True, position = 'test', N_grid = 400, mini_grid = 40, sigma_grid = 100, x_off = 0, y_off = 14, z_off = -11, fourier_smear = 0.25, offset = 0.0, comment = 'Default case', offset_vector = np.array([[0,0,1]]), default_s_q =0.669314655271, q_val = 6.28, q_tol = 0.015, total_steps = 40, timestep = 0.005, I_Q = False, S_Q = False, ISF = False, smeared_integral = False, direct_point = False, atoms_add = None, frames =40 ):
        if (int(I_Q) + int(S_Q) + int(ISF)) != 1:
            raise TypeError('Choose one and only one computation item')
        if (int(smeared_integral) + int(direct_point)) != 1:
            raise TypeError('Choose between direct computation and smeared integral')
        self.box_length = box_length
        self.smear = smear
        self.position = position
        self.N_grid = N_grid
        self.mini_grid = mini_grid
        self.sigma_grid = sigma_grid
        self.density_cutoff = int(np.floor(5.0*self.N_grid/self.sigma_grid))
        self.fourier_smear = fourier_smear
        if rc < 4/fourier_smear:
            self.rc = 4.0/self.fourier_smear
        else:
            self.rc = rc
        self.rv = 1.02*self.rc
        self.skin = self.rv - self.rc
        self.repeat_count = int(np.ceil(2*self.rv/self.box_length))
        self.x_global = self.N_grid//2 + x_off
        self.y_global = self.N_grid//2 + y_off
        self.z_global = self.N_grid//2 + z_off
        self.x0 = self.mini_grid + x_off
        self.y0 = self.mini_grid + y_off
        self.z0 = self.mini_grid + z_off
        self.tol = 1e-3
        self.offset = offset
        self.comment = comment
        self.offset_vector = offset_vector
        self.default_s_q = default_s_q
        self.q_val = q_val
        self.q_tol = q_tol
        self.filename = filename
        self.total_steps = total_steps
        self.timestep = timestep
        self.S_Q = S_Q
        self.I_Q = I_Q
        self.ISF = ISF
        self.smeared_integral = smeared_integral
        self.direct_point = direct_point
        self.atoms_add = atoms_add
        self.frames = frames

    def generate_configuration(self):
        if self.position == 'test':
            N_atoms = 4
            pos = self.offset*self.N_grid*np.repeat(np.array([[0,0,1]]),N_atoms,axis = 0) \
                + self.box_length/2 \
                + np.array([[1,0,0],[0,1,0],[0,0,1],[1/3-np.sqrt(10)/3,1/3-np.sqrt(10)/3,1/3-np.sqrt(10)/3]])
            hin = np.array([[0.0,self.box_length],[0.0,self.box_length],[0.0,self.box_length]])
        elif self.position == 'single':
            N_atoms = 1
            pos = self.box_length*np.array([[2.0,1.0,0.6]])
            hin = np.array([[0.0,self.box_length],[0.0,self.box_length],[0.0,self.box_length]])
        elif self.position == 'file':
            pos, hins, N_atoms = load_dumpfile_atom_data(self.filename, 1, 1, verbose=False, h_full=False)
            hin = hins[:,:,0]
            self.box_length=np.mean(hin[:,1]-hin[:,0])
        elif self.position == 'time':
            if self.total_steps < 100:
                pos, hins, N_atoms = load_dumpfile_atom_data(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            else:
                pos, hins, N_atoms = load_dumpfile_atom_data_fast(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            hin = hins[:,:,0]
            self.box_length=np.mean(hin[:,1]-hin[:,0])
        if np.linalg.norm(np.diff(hin,axis=1)-self.box_length) > 1e-8:
            raise TypeError('The simulation box needs to be cubic')
        s = pos/self.box_length
        s = s - 0.5
        s = s-np.round(s)
        pos = self.box_length*(s+0.5)
        if self.position == 'time':
            if self.atoms_add is None:
                atoms_add = N_atoms
            else:
                atoms_add = self.atoms_add
            pos_input = np.zeros((atoms_add+2,3,self.total_steps))
            for i in range(self.total_steps):
                temp_pos = pos[i*atoms_add:(i+1)*atoms_add,:]
                pos_input[:2,:,i] = hin.T
                pos_input[2:,:,i] = temp_pos
            self.pos_input = pos_input
        self.pos = pos
        self.hin = hin
        self.N_atoms = N_atoms
    def generate_q_grid(self):
        x_grid = np.linspace(self.N_grid//2-self.mini_grid,self.N_grid//2+self.mini_grid,2*self.mini_grid+1,dtype=np.int)
        x_grid_global = np.linspace(0,self.N_grid-1,self.N_grid)*self.box_length/self.N_grid
        [X0,Y0,Z0] = np.meshgrid(x_grid_global,x_grid_global,x_grid_global)
        mini_grid_index = np.ix_(x_grid,x_grid,x_grid)
        q_grid = 2*np.pi*np.linspace(-(self.N_grid//2),self.N_grid//2-1,self.N_grid,dtype=np.int)/self.box_length
        [Qx, Qy, Qz] = np.meshgrid(q_grid,q_grid,q_grid)
        Q_line = np.zeros((self.N_grid**3,3))
        Q_line[:,0] = Qx.reshape(-1,1).flatten()
        Q_line[:,1] = Qy.reshape(-1,1).flatten()
        Q_line[:,2] = Qz.reshape(-1,1).flatten()
        q_grid_magnitude = np.linalg.norm(Q_line,axis=1).reshape(Qx.shape)
        q_probe = q_grid_magnitude[mini_grid_index]
        probe_index = np.where(abs(q_probe.reshape(-1,1)-self.q_val)<self.q_tol)[0]
        q_values = len(x_grid)
        sigma_weights = self.fourier_smear/(2*np.pi)
        smear_width = int(np.ceil(6*sigma_weights*self.box_length))
        [X_grid,Y_grid,Z_grid]=np.meshgrid(x_grid,x_grid,x_grid)
        if self.position == 'time':
            point_grid_x = x_grid[self.x0-smear_width:self.x0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            point_grid_y = x_grid[self.y0-smear_width:self.y0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            point_grid_z = x_grid[self.z0-smear_width:self.z0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            dummy_grid = np.zeros(X_grid.shape)
            dummy_ind =np.ix_(point_grid_y,point_grid_x,point_grid_z)
            dummy_grid[dummy_ind] = 1.0
            point_grid = np.where(dummy_grid.reshape(-1,1)==1.0)[0]
            self.point_grid = point_grid
        scale_pos = np.zeros((3,3))
        h = self.hin
        scale_pos[:2,:] = h.T
        scale_pos[2,:] = self.box_length*np.array([0.0,0.0,0.0])
        smear_adjust = s_q_from_pos_smear(scale_pos,self.N_grid, self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True, structure_factor=True, correction = True)
        qx = 2*np.pi*(X_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qy = 2*np.pi*(Y_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qz = 2*np.pi*(Z_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qz_array = 2*np.pi*(Z_grid[self.y0,self.x0,:]-self.N_grid//2)/self.box_length
        q3_point=np.array([[qx,qy,qz]])
        q3_array = np.zeros((q_values,3))
        q3_array[:,0] = q3_point[0,0]*np.ones(q_values)
        q3_array[:,1] = q3_point[0,1]*np.ones(q_values)
        q3_array[:,2] = qz_array
        q_magnitude_array = np.linalg.norm(q3_array,axis=1).flatten()  #Scaling q magnitude along the line in q-space
        q_magnitude = np.linalg.norm(q3_point,axis=1).flatten()  # Scaling q magnitude at chosen value
        xind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        yind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        zind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        [x_weight_grid,y_weight_grid,z_weight_grid]=np.meshgrid(x_grid[xind],x_grid[yind],x_grid[zind])
        grid_center = np.array([X_grid[self.mini_grid,self.mini_grid,self.mini_grid],Y_grid[self.mini_grid,self.mini_grid,self.mini_grid],Z_grid[self.mini_grid,self.mini_grid,self.mini_grid]])
        weights = norm3d(grid_center/self.box_length,x_weight_grid/self.box_length,y_weight_grid/self.box_length,z_weight_grid/self.box_length,sigma_weights,self.N_grid//2,self.N_grid//2,self.N_grid//2)
        weights = np.reshape(weights,(-1,1))
        self.q_values = q_values
        self.qz_array = qz_array
        self.q3_point = q3_point
        self.q3_array = q3_array
        self.q_magnitude = q_magnitude
        self.q_magnitude_array = q_magnitude_array
        self.weights = weights
        self.smear_width = smear_width
        self.probe_index = probe_index
        self.q_grid_magnitude = q_grid_magnitude
        self.rescale_factor = smear_adjust
    def real_space_s_q_pairwise(self):
        h = np.diag(self.box_length*np.ones(3))
        if self.position != 'time':
            pos_ext, h_ext = config_repeater(self.pos,h,size=self.repeat_count)
            hinv=np.linalg.inv(h_ext)
            nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
            pos_ref = pos_ext
            r_array = get_r_array(pos_ext, h_ext, self.rc, nnlist=index, atoms_add = self.N_atoms)
            r_sq=np.linalg.norm(r_array,axis=1)
            rho = self.N_atoms/np.linalg.det(h)
            #if self.fourier_smear < 1/4.0:
                #set_num_threads(8)
            s_pairwise = s_q3_from_pos_par(self.q3_array, r_array, self.rc, rho, self.N_atoms,  smear = self.smear, ddq = self.fourier_smear , r_sq=r_sq)
            if self.S_Q:
                self.s_pairwise = s_pairwise.real
            elif self.I_Q:
                self.s_pairwise = s_pairwise.real*form_factor_analytical(self.q_magnitude_array)**2
        else:
            s_pairwise = np.zeros(self.total_steps)
            for i in range(self.total_steps):
                pos_time = self.pos[i*self.N_atoms:(i+1)*self.N_atoms]
                pos_ext, h_ext = config_repeater(pos_time,h,size=self.repeat_count)
                hinv=np.linalg.inv(h_ext)
                if i==0:
                    nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
                    pos_ref = pos_ext
                else:
                    if np.max(np.linalg.norm(pbc(pos_ext-pos_ref,h_ext,hinv),axis=1))>self.skin/2:
                        nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
                        pos_ref = pos_ext
                r_array = get_r_array(pos_ext, h_ext, self.rc, nnlist=index, atoms_add = self.N_atoms)
                r_sq=np.linalg.norm(r_array,axis=1)
                rho = self.N_atoms/np.linalg.det(h)
                #set_num_threads(8)
                s_pairwise[i] = s_q3_from_pos_par(self.q3_point, r_array, self.rc, rho, self.N_atoms,  smear = self.smear, ddq = self.fourier_smear , r_sq=r_sq)
            if self.S_Q:
                self.s_pairwise = s_pairwise.real
            elif self.I_Q:
                self.s_pairwise = s_pairwise.real*form_factor_analytical(self.q_magnitude)**2

    def real_space_s_q_position(self):
        s_position= np.zeros(self.q_values,dtype = np.complex)
        for j in range(self.q_values):
            q3 = self.q3_array[j,:]
            q = np.linalg.norm(q3)
            adj_ISF = np.exp(-1j*np.dot(self.pos,q3.T)).sum()
            s_position[j]=adj_ISF*np.conj(adj_ISF)/self.N_atoms

        if self.S_Q:
            self.s_position = s_position.real
        elif self.I_Q:
            self.s_position = s_position.real*form_factor_analytical(self.q_magnitude_array)**2

    def fourier_space(self):
        s_smear = np.zeros(self.q_values,dtype = np.complex)
        h_pos = np.zeros((self.N_atoms+2,3))
        h = self.hin
        h_pos[:2,:] = h.T
        h_pos[2:,:] = self.pos
        grid_s_q = s_q_from_pos_smear(h_pos,self.N_grid, self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True, structure_factor=True, correction_grid = self.rescale_factor, q_magnitude = self.q_grid_magnitude)
        s_fourier = (grid_s_q[self.y_global,
                              self.x_global,
                              self.N_grid//2-self.mini_grid:self.N_grid//2+self.mini_grid+1] \
                              ).astype(np.complex)
        if self.S_Q:
            self.s_fourier = s_fourier.real
        elif self.I_Q:
            self.s_fourier = s_fourier.real*form_factor_analytical(self.q_magnitude_array)**2
        for i in range(self.q_values):
            s_q_integration_grid = np.zeros((2*self.smear_width+1)**3)
            xind = np.linspace(self.x_global-self.smear_width,self.x_global+self.smear_width,2*self.smear_width+1,dtype=np.int)
            yind = np.linspace(self.y_global-self.smear_width,self.y_global+self.smear_width,2*self.smear_width+1,dtype=np.int)
            zind = np.linspace(self.N_grid//2-self.mini_grid+i-self.smear_width,self.N_grid//2-self.mini_grid+i+self.smear_width,2*self.smear_width+1,dtype=np.int)
            fill_ind  = np.ix_(yind,xind,zind)
            s_q_integration_grid = np.reshape(grid_s_q[fill_ind],(-1,1))
            integrated_array = np.dot(np.diag(self.weights.flatten()),s_q_integration_grid)*self.box_length**-3
            s_smear[i] = np.sum(integrated_array,axis=0)
        if self.S_Q:
            self.s_fourier_smear = s_smear.real
        elif self.I_Q:
            self.s_fourier_smear = s_smear.real*form_factor_analytical(self.q_magnitude_array)**2
    def fourier_space_time(self):
        pool=mp.Pool(mp.cpu_count())
        if self.smeared_integral:
            compute_partial = partial(s_q_from_pos_smear,N = self.N_grid, wdt = self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True,structure_factor=True, correction_grid = self.rescale_factor, q_magnitude = self.q_grid_magnitude, ind_need = self.point_grid)
        elif self.direct_point:
            compute_partial = partial(s_q_from_pos_smear,N = self.N_grid, wdt = self.sigma_grid,
                         cs = self.density_cutoff, ms = self.mini_grid, dump=True, ISF=self.ISF, intensity =self.I_Q, structure_factor=self.S_Q, correction_grid = self.rescale_factor, q_magnitude = self.q_grid_magnitude, ind_need = self.point_grid)
        s_time = pool.map(compute_partial,[self.pos_input[:,:,r] for r in range(self.total_steps)])
        s_time = np.asarray(s_time)
        s_time = s_time.T.real
        if self.smeared_integral:
            integrated_array = (self.I_Q*(form_factor_analytical(self.q_magnitude)**2-1.0)+1.0)*np.dot(np.diag(self.weights.flatten()),s_time)*self.box_length**-3
            s_time_array = np.sum(integrated_array,axis=0)
            self.s_fourier_time = s_time_array
        elif self.direct_point:
            self.s_fourier_time = s_time

    def test_consistency(self, plot_results = False, fig_id = 1):
        self.generate_configuration()
        self.generate_q_grid()
        self.real_space_s_q_pairwise()
        self.real_space_s_q_position()
        self.fourier_space()
        print()
        print('*'*50)
        print('Test condition: %s'%self.comment)
        print('*'*50)
        print()
        print('The s(q) values obtained:\n'
              '1) From the real space method (position): %.12f\n'
              '2) From the real space method (pairwise): %.12f\n'
              '3) From the fourier space method:         %.12f\n'
              '4) From the fourier space method (smear): %.12f'
              %(self.s_position[self.z0],self.s_pairwise[self.z0],self.s_fourier[self.z0],self.s_fourier_smear[self.z0]))
        if self.position == 'test':
            error_pos_four = abs((self.default_s_q-self.s_fourier[self.z0])/self.default_s_q)
            error_pair_four = abs((self.default_s_q-self.s_position[self.z0])/self.default_s_q)
        else:
            error_pos_four = abs((self.s_position-self.s_fourier)/self.s_position)
            error_pair_four = abs((self.s_pairwise-self.s_fourier_smear)/self.s_pairwise)
        cleared_count = int(error_pos_four.mean()  < self.tol) \
                      + int(error_pair_four.mean() < self.tol)
        if cleared_count == 2:
            print('TEST: '+bcolors.GRN+'PASSED'+bcolors.RESET )
            test_success = True
        else:
            print('TEST: '+bcolors.RED+'FAILED'+bcolors.RESET )
            test_success = False
        if plot_results:
            plt.figure(fig_id)
            plt.plot(self.qz_array, self.s_position,     label = 'Real space - position')
            plt.plot(self.qz_array, self.s_pairwise, ':',label = 'Real space - pairwise')
            plt.plot(self.qz_array, self.s_fourier,  'o',label = 'Fourier space')
            plt.plot(self.qz_array, self.s_fourier_smear,  '*',label = 'Fourier space - smeared')
            plt.legend(loc='best')
        return test_success
    def compute_results(self):
        self.generate_configuration()
        self.generate_q_grid()
        self.real_space_s_q_pairwise()
        self.real_space_s_q_position()
        self.fourier_space()
    def compute_and_save(self):
        self.generate_configuration()
        self.generate_q_grid()
        if self.position == 'file':
            self.fourier_space()
            self.real_space_s_q_position()
            self.real_space_s_q_pairwise()
            np.savetxt('gg_cor_%1.2f.txt'%(self.fourier_smear),self.s_pairwise)
            np.savetxt('g_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier_smear)
            np.savetxt('xx_cor_%1.2f.txt'%(self.fourier_smear),self.s_position)
            np.savetxt('x_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier)
        elif self.position =='time':
            self.fourier_space_time()
            np.savetxt('s_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier_time)
            self.real_space_s_q_pairwise()
            np.savetxt('ss_cor_%1.2f.txt'%(self.fourier_smear),self.s_pairwise)
    def time_auto_correlation(self):
        if self.position != 'time':
            raise TypeError('Position type error: Valid only for time')
        pool=mp.Pool(mp.cpu_count())
        correl = partial(auto_corr,frames=self.frames)
        items = self.s_fourier_time.shape[0]
        g2 = pool.map(correl, [self.s_fourier_time[i,:] for i in range(items)])
        g2 = np.asarray(g2)
        beta=g2[:,0]-1.0
        g2_exp_fit = np.dot(np.diag(1.0/(beta)),(g2-1.0))
        g2_exp_fit_mean = np.mean(g2_exp_fit,axis=0)
        g2_mean = np.mean(g2,axis=0)
        self.g2 = g2
        self.g2_exp_fit = g2_exp_fit
        self.g2_mean = g2_mean
        self.g2_exp_fit_mean = g2_exp_fit_mean
    def compute_and_save_correlation(self):
        self.generate_configuration()
        self.generate_q_grid()
        self.fourier_space_time()
        self.time_auto_correlation()
        np.savez('ts_cor_%.2f_%.2f.npz'%(self.q_val,self.atoms_add),b = self.g2)
        np.savez('fit_cor_%.2f_%.2f.npz'%(self.q_val,self.atoms_add),b = self.g2_exp_fit)
        np.savetxt('ts_cor_%.2f_%.2f.txt'%(self.q_val,self.atoms_add), self.g2_mean)
        np.savetxt('fit_cor_%.2f_%.2f.txt'%(self.q_val,self.atoms_add), self.g2_exp_fit_mean)

class XPCS_Suite():
    def __init__(self, filename = 'dump.ljtest', q_val = 6.28, frames = 33,
                 pool = None, atoms_add = None, sigma_grid = 400, ISF = False,
                 N_grid = 400, mini_grid = 40, density_cutoff = 5, q_tol = 0.05,
                 intensity =  True, total_steps = 5000, timestep = 0.05,
                 system = 'liquid', atom_type = None, dir_name = ''):
        self.filename = filename
        self.q_val = q_val
        self.frames = frames
        self.pool = pool
        self.atoms_add = atoms_add
        self.sigma_grid = sigma_grid
        self.N_grid = N_grid
        self.mini_grid = mini_grid
        self.density_cutoff = density_cutoff
        self.q_tol = q_tol
        self.intensity = intensity
        self.total_steps = total_steps
        self.ISF = ISF
        self.timestep = timestep
        self.atom_type = atom_type
        self.system = system
        self.dir_name = dir_name
        self.times = np.linspace(0, self.timestep * (self.frames - 1), self.frames)
        if system == 'liquid':
            self.nondim_t = 2.156e-12
            self.nondim_d = 3.405e-10
        elif system == 'DPN':
            if self.atom_type is None:
                raise TypeError("atom_type must be assigned out of ligand or metal: atom_type = \'metal\'")
            self.nondim_t = 2.3e-10
            self.nondim_d = 15e-10
        self.times_ps = self.times / self.nondim_t * 1e12
        self.times_fs = self.times / self.nondim_t * 1e15

    def load_and_process_trajectory(self):
        if self.system == 'liquid':
            if self.total_steps < 100:
                self.pos, self.hins, self.N_atoms = load_dumpfile_atom_data(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            else:
                self.pos, self.hins, self.N_atoms = load_dumpfile_atom_data_fast(self.filename, self.total_steps, 1, verbose=False, h_full=False)
        elif self.system == 'DPN':
            self.pos_a, self.pos_b, self.hins, self.atoms_a, self.atoms_b = load_dumpfile_atom_data_binary_fast(self.filename, self.total_steps,
                                                                                      1, verbose=False,
                                                                                      family='polymer', h_full=False)
            if self.atom_type == 'metal':
                self.pos = self.pos_a.copy()
                self.N_atoms = self.atoms_a
            elif self.atom_type == 'ligand':
                self.pos = self.pos_b.copy()
                self.N_atoms = self.atoms_b
        self.hin = self.hins[:, :, 0]
        self.box_length = np.mean(self.hin[:, 1] - self.hin[:, 0])
        s = self.pos / self.box_length
        s = s - 0.5
        s = s - np.round(s)
        self.pos = self.box_length * (s + 0.5)
        self.h = self.hin
        if self.atoms_add is None:
            self.atoms_add = self.N_atoms
        else:
            self.atoms_add = self.atoms_add
        self.pos_input = np.zeros((self.atoms_add + 2, 3, self.total_steps))
        for i in range(self.total_steps):
            temp_pos = self.pos[i * self.N_atoms:i * self.N_atoms + self.atoms_add, :]
            self.pos_input[:2, :, i] = self.hin.T
            self.pos_input[2:, :, i] = temp_pos

    def prepare_intensity_correction(self):
        self.scale_pos = np.zeros((3, 3))
        self.scale_pos[:2, :] = self.h.T
        self.scale_pos[2, :] = self.box_length * np.array([0.0, 0.0, 0.0])
        if self.intensity:
            self.rescale_factor = s_q_from_pos_smear(self.scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                            cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                            correction=True)
        if self.ISF:
            self.rescale_factor_ISF = s_q_from_pos_smear(self.scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                                cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True, correction=True)

    def prepare_fourier_grid(self):
        x_grid = np.linspace(self.N_grid // 2 - self.mini_grid, self.N_grid // 2 + self.mini_grid, 2 * self.mini_grid + 1, dtype=np.int)
        mini_grid_index = np.ix_(x_grid, x_grid, x_grid)
        q_grid = 2 * np.pi * np.linspace(-(self.N_grid // 2), self.N_grid // 2 - 1, self.N_grid, dtype=np.int) / self.box_length
        [Qx, Qy, Qz] = np.meshgrid(q_grid, q_grid, q_grid)
        Q_line = np.zeros((self.N_grid ** 3, 3))
        Q_line[:, 0] = Qx.reshape(-1, 1).flatten()
        Q_line[:, 1] = Qy.reshape(-1, 1).flatten()
        Q_line[:, 2] = Qz.reshape(-1, 1).flatten()
        self.q_grid_magnitude = np.linalg.norm(Q_line, axis=1).reshape(Qx.shape)
        self.q_probe = self.q_grid_magnitude[mini_grid_index]
        if self.intensity:
            self.correct_grid = self.rescale_factor[mini_grid_index].reshape(-1, 1)
        if self.ISF:
            self.correct_grid_ISF = self.rescale_factor_ISF[mini_grid_index].reshape(-1, 1)
        self.probe_index = np.where(abs(self.q_probe.reshape(-1, 1) - self.q_val) < self.q_tol)[0]
        if len(self.probe_index) == 0:
            raise TypeError('No points selected')

    def solve_intensity(self):
        if self.intensity:
            compute_partial = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                      cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                      q_magnitude=self.q_grid_magnitude, ind_need=self.probe_index)
            self.s_time = self.pool.map(compute_partial, [self.pos_input[:, :, r] for r in range(self.total_steps)])
            self.s_time = np.asarray(self.s_time)
            self.s_time = self.s_time.T.real
            self.s_time = np.dot(np.diag(self.correct_grid[self.probe_index].flatten()), self.s_time)
        if self.ISF:
            compute_partial_ISF = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                          cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True,
                                          q_magnitude=self.q_grid_magnitude, ind_need=self.probe_index)
            self.s_time_ISF = self.pool.map(compute_partial_ISF, [self.pos_input[:, :, r] for r in range(self.total_steps)])
            self.s_time_ISF = np.asarray(self.s_time_ISF)
            self.s_time_ISF = self.s_time_ISF.T.real
            self.s_time_ISF = np.dot(np.diag(self.correct_grid_ISF[self.probe_index].flatten()), self.s_time_ISF)

    def solve_correlations(self):
        if self.intensity:
            self.items = self.s_time.shape[0]
            correl = partial(auto_corr, frames=self.frames)
            self.g2 = self.pool.map(correl, [self.s_time[i, :] for i in range(self.items)])
            self.g2 = np.asarray(self.g2)
            self.beta = self.g2[:, 0] - 1.0
            self.g2_exp_fit = np.dot(np.diag(1.0 / (self.beta)), (self.g2 - 1.0))
            self.g2_exp_fit_mean = np.mean(self.g2_exp_fit, axis=0)
            self.g2_exp_fit_mean = self.g2_exp_fit_mean - self.g2_exp_fit_mean.min() + 1e-6
            self.g2_mean = np.mean(self.g2, axis=0)
        if self.ISF:
            self.items = self.s_time_ISF.shape[0]
            correl_ISF = partial(ISF_corr, frames=self.frames)
            self.f_qt = self.pool.map(correl_ISF, [self.s_time_ISF[i, :] for i in range(self.items)])
            self.f_qt = np.asarray(self.f_qt)
            self.F_qt = np.mean(self.f_qt, axis=0) ** 2

    def save_correlations(self):
        if self.intensity:
            np.savez(self.dir_name+'ts_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.g2.real)
            np.savez(self.dir_name+'s_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.s_time.real)
            np.savez(self.dir_name+'fit_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.g2_exp_fit.real)
            np.savetxt(self.dir_name+'ts_cor_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.g2_mean.real)
            np.savetxt(self.dir_name+'Auto_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.g2_exp_fit_mean.real)
        if self.ISF:
            np.savetxt(self.dir_name+'ISF_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.F_qt)

    def contrast_computation(self):
        tots = [1, 10, 100, 500, 1000, 2000, 5000]
        tots_ps = np.array(tots) * self.timestep / self.nondim_t * 1e12
        tots_fs = np.array(tots) * self.timestep / self.nondim_t * 1e15
        st_array = []
        for i, tot in enumerate(tots):
            s_t = self.s_time[:, :tot].mean(1)
            st_array.append([tots_ps[i], optical_contrast(s_t)])

        self.contrast_data = np.array(st_array)

    def test_class(self):
        g2_func = np.load('ts_cor_%.2f_%d_1000.npz' % (self.q_val, self.atoms_add))['b']
        if np.allclose(self.g2,g2_func):
            print("Class carries out the same computation")
        else:
            print("Class fails the same computation")

    def test_class_workflow(self):
        self.load_and_process_trajectory()
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()
        self.solve_correlations()
        self.test_class()

    def get_gamma(self):
        self.times = np.linspace(0,self.timestep*(self.frames-1),self.frames)
        if self.atoms_add < self.N_atoms // 10:
            popt, _ = curve_fit(objective_lin,self.times[3:19], np.log(self.g2_exp_fit_mean[3:19]))
        else:
            popt, _ = curve_fit(objective_lin, self.times[2:6], np.log(self.g2_exp_fit_mean[2:6]))
        a, b  = popt
        self.gamma_ns = -0.5*a*1e-9/self.nondim_t

    def store_gamma_workflow(self):
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()
        self.solve_correlations()
        self.get_gamma()

    def get_intensity_workflow(self):
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()


def end_to_end_pos_to_g2(pos, q_val, frames, pool, save_file = 'sample', atoms_add = 40, h_in = None, total_atoms = 200):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            pos : float, dimension (natoms,3)
                atoms in one array -- used in multiprocessing
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            pool : multiprocessing object,
                Pool of cpus for multiprocessing
            q_val : float, dimension (N-q, 3)
                Array of all wave-vectors for form factor computation
            save_file : str
                Name of npz file to save s(q), g2(q,t) and F(q) array to
            atoms_add :  int
                Number of atoms to compute the XPCS signal over
            h_in : float, dimension (3,2)
                hi and lo values in x, y, and z direction
            total_atoms : int
                If position array doesn't encode all atoms

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    if len(pos.shape) == 3:
        total_steps = pos.shape[2]
        N_atoms = pos.shape[0]
        pos = pos.reshape(pos.shape[0]*pos.shape[2],pos.shape[1])
    else: 
        N_atoms = total_atoms
        total_steps = pos.shape[0]//N_atoms
    if h_in is None:
        boxl = 17.0
    else:
        boxl = h_in
    q3_eff = q_val ** 2
    t_size = 0.05 
    lags = np.linspace(0, t_size * (total_steps - 1), total_steps)
    sigma_grid = 400
    N_grid = 400
    mini_grid = 40
    density_cutoff = 5
    q_tol = 0.05
    I_Q = True
    position = 'time'
    s = pos / boxl
    s = s - 0.5
    s = s - np.round(s)
    pos = boxl * (s + 0.5)
    scale_pos = np.zeros((3, 3))
    h = np.array([[0.0,boxl],[0.0,boxl],[0.0,boxl]])
    scale_pos[:2, :] = h.T
    scale_pos[2, :] = boxl * np.array([0.0, 0.0, 0.0])
    rescale_factor = s_q_from_pos_smear(scale_pos, N=N_grid, wdt=sigma_grid,
                                        cs=density_cutoff, ms=mini_grid, dump=True, structure_factor=True,
                                        correction=True)
    rescale_factor_ISF = s_q_from_pos_smear(scale_pos, N=N_grid, wdt=sigma_grid,
                                            cs=density_cutoff, ms=mini_grid, dump=True, ISF=True, correction=True)
    if position == 'time':
        if atoms_add is None:
            atoms_add = N_atoms
        else:
            atoms_add = atoms_add
        pos_input = np.zeros((atoms_add + 2, 3, total_steps))
        for i in range(total_steps):
            temp_pos = pos[i * N_atoms:i * N_atoms + atoms_add, :]
            pos_input[:2, :, i] = h.T
            pos_input[2:, :, i] = temp_pos
    x_grid = np.linspace(N_grid // 2 - mini_grid, N_grid // 2 + mini_grid, 2 * mini_grid + 1, dtype=np.int)
    mini_grid_index = np.ix_(x_grid, x_grid, x_grid)
    q_grid = 2 * np.pi * np.linspace(-(N_grid // 2), N_grid // 2 - 1, N_grid, dtype=np.int) / boxl
    [Qx, Qy, Qz] = np.meshgrid(q_grid, q_grid, q_grid)
    Q_line = np.zeros((N_grid ** 3, 3))
    Q_line[:, 0] = Qx.reshape(-1, 1).flatten()
    Q_line[:, 1] = Qy.reshape(-1, 1).flatten()
    Q_line[:, 2] = Qz.reshape(-1, 1).flatten()
    q_grid_magnitude = np.linalg.norm(Q_line, axis=1).reshape(Qx.shape)
    q_probe = q_grid_magnitude[mini_grid_index]
    correct_grid = rescale_factor[mini_grid_index].reshape(-1, 1)
    correct_grid_ISF = rescale_factor_ISF[mini_grid_index].reshape(-1, 1)
    probe_index = np.where(abs(q_probe.reshape(-1, 1) - q_val) < q_tol)[0]
    if len(probe_index) == 0:
        raise TypeError('No points selected')
    compute_partial = partial(s_q_from_pos_smear, N=N_grid, wdt=sigma_grid,
                              cs=density_cutoff, ms=mini_grid, dump=True, structure_factor=True,
                              q_magnitude=q_grid_magnitude, ind_need=probe_index)
    s_time = pool.map(compute_partial, [pos_input[:, :, r] for r in range(total_steps)])
    s_time = np.asarray(s_time)
    s_time = s_time.T.real
    s_time = np.dot(np.diag(correct_grid[probe_index].flatten()), s_time)
    compute_partial_ISF = partial(s_q_from_pos_smear, N=N_grid, wdt=sigma_grid,
                                  cs=density_cutoff, ms=mini_grid, dump=True, ISF=True, q_magnitude=q_grid_magnitude,
                                  ind_need=probe_index)
    s_time_ISF = pool.map(compute_partial_ISF, [pos_input[:, :, r] for r in range(total_steps)])
    s_time_ISF = np.asarray(s_time_ISF)
    s_time_ISF = s_time_ISF.T.real
    s_time_ISF = np.dot(np.diag(correct_grid_ISF[probe_index].flatten()), s_time_ISF)
    correl = partial(auto_corr, frames=frames)
    correl_ISF = partial(ISF_corr, frames=frames)
    items = s_time.shape[0]
    g2 = pool.map(correl, [s_time[i, :] for i in range(items)])
    g2 = np.asarray(g2)
    f_qt = pool.map(correl_ISF, [s_time_ISF[i, :] for i in range(items)])
    f_qt = np.asarray(f_qt)
    F_qt = np.mean(f_qt, axis=0) ** 2
    beta = g2[:, 0] - 1.0
    g2_exp_fit = np.dot(np.diag(1.0 / (beta)), (g2 - 1.0))
    g2_exp_fit_mean = np.mean(g2_exp_fit, axis=0)
    g2_mean = np.mean(g2, axis=0)
    np.savez('%s_%.2f_%d.npz' % (save_file, q_val, atoms_add), g2_m = g2_mean, g2_em = g2_exp_fit_mean, F = F_qt)

def write_lammps_dump(posit, save_file = 'sample', h_in = None, total_atoms = 200):
    '''Write a lammps dumpfile from a position array

                    Parameters
                    ----------
                    posit : float, dimension (natoms, 3, steps)
                        Atomic positions over all frames
                    save_file : str,
                        dump_filename to save to
                    h_in : float
                        Assumin cubic box -- just one side of the box


    '''
    if h_in is None:
        boxl = 17.0
    else:
        boxl = h_in
    if len(posit.shape) == 3:
        total_steps = posit.shape[2]
        N_atoms = posit.shape[0]
        for i in range(total_steps):
            s = posit[:,:,i] / boxl
            s = s - 0.5
            s = s - np.round(s)
            posit[:,:,i] = (s + 0.5)
        fid = open('dump.%s'%save_file,'w')
        for i in range(total_steps):
            fid.write('ITEM: TIMESTEP\n')
            fid.write('%d\n'%(i))
            fid.write('ITEM: NUMBER OF ATOMS\n')
            fid.write('%d\n'%(N_atoms))
            fid.write('ITEM: BOX BOUNDS pp pp pp\n')
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('ITEM: ATOMS id type xs ys zs\n')
            for j in range(N_atoms):
                fid.write('%d 1 %.5f %.5f %.5f\n'%(j+1,posit[j,0,i],posit[j,1,i],posit[j,2,i]))
        fid.close()
    else: 
        N_atoms = total_atoms
        total_steps = posit.shape[0]//N_atoms
        s = posit / boxl
        s = s - 0.5
        s = s - np.round(s)
        posit = (s + 0.5)
        fid = open('dump.%s'%save_file,'w')
        for i in range(total_steps):
            fid.write('ITEM: TIMESTEP\n')
            fid.write('%d\n'%(i))
            fid.write('ITEM: NUMBER OF ATOMS\n')
            fid.write('%d\n'%(N_atoms))
            fid.write('ITEM: BOX BOUNDS pp pp pp\n')
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('ITEM: ATOMS id type xs ys zs\n')
            for j in range(N_atoms):
                fid.write('%d 1 %.5f %.5f %.5f\n'%(j+1,posit[i*N_atoms+j,0],posit[i*N_atoms+j,1],posit[i*N_atoms+j,2]))
        fid.close()

