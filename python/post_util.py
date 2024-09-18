import sys, os
#sys.path.append(os.path.realpath(os.environ['DPNgit']+'/python'))
from md_util import *
import re
import ase
from ase.io import read, write

def make_lj(read_file = 'dump.ljnd', save_file = 'dump.lj'):
    '''Convert LJ units to metal units (Liquid Ar)

        Parameters
        ----------
        read_file : string
            Filename string, for data in LJ units
        save_file : string
            Filename string, for data in metal units
        Returns
        -------
        Data file in metal units
    '''
    lines = open(read_file,'r').readlines()
    fid = open(save_file,'w')
    for i in range(5):
        fid.write(lines[i])

    a = np.array(lines[5].split()[:2],dtype=float)
    fid.write('%f %f\n'%(3.4*a[0],3.4*a[1]))
    a = np.array(lines[6].split()[:2],dtype=float)
    fid.write('%f %f\n'%(3.4*a[0],3.4*a[1]))
    a = np.array(lines[7].split()[:2],dtype=float)
    fid.write('%f %f\n'%(3.4*a[0],3.4*a[1]))
    fid.write(lines[8])
    n_at = np.genfromtxt(read_file,skip_header=3,dtype = int,max_rows=1).item(0)
    dd = 3.4/2156
    for i in range(9,n_at+9):
        a = np.array(lines[i].split()[2:],dtype=float)
        b = np.array(lines[i].split()[:2],dtype=int)
        fid.write('%d %d %f %f %f %f %f %f\n'%(b[0],b[1],3.4*a[0],3.4*a[1],3.4*a[2], dd*a[3],dd*a[4],dd*a[5]))
        #fid.write('%d %d %f %f %f\n'%(b[0],b[1],3.4*a[0],3.4*a[1],3.4*a[2]))

    fid.close()

def load_custom(filename, skip_to_np=2, skip_to_box=4, skip_to_pos=11):
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
    
    nparticles = np.genfromtxt(filename, skip_header=skip_to_np,  dtype=np.int,    max_rows=1).item(0)
    box        = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:,:2]
    h=box
    rawdata    = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata=rawdata[np.argsort(rawdata[:,0])]
    pos = np.zeros([nparticles, 3])
    pos[rawdata[:,0].astype(int)-1] = rawdata[:,2:5]

    return pos, h

def read_CNA(filename,steps):
    '''Load LAMMPS .data type configuration files

        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        steps : int
            Number of steps for which Coordination number analysis results are available

        Returns
        -------
        CNA : ndarray (Natoms, 3, steps)
            CNA of atoms per step
    '''
    lines = open(filename,'r').readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[3].strip()),dtype=int)
    nat = npar[0]
    CNA = np.zeros((nat,3,steps),dtype=float)
    for j in range(steps):
        ads = 9 + (nat+9)*j
        for i in range(nat):
            a = lines[i+ads].split()
            CNA[i,:,j] = np.array([int(a[0]),int(a[1]),float(a[2])])
        CNA[:,:,j] = CNA[np.argsort(CNA[:,0,j]),:,j]
    return CNA

def gr_GNN(filename,bins = None ,rc = None, nd = None, database=False):
    '''RDF computing function

        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        rc : float,
            Cutoff radius for g(r)
        bins : int
            Number of bins
        nd : float,
            non-dimensional distance w.r.t metal units

        Returns
        -------
        g_array: float, dimension (bins,)
            Normalized histograms to be equivalent to the g(r)
    '''
    p, ho = load_custom(filename,2,4,11)
    bl = np.mean(ho[:,1]-ho[:,0])
    s = p/bl
    s = s - 0.5
    s = s - np.round(s)
    p = bl*(s+0.5)
    if database:
        return p/nd, ho/nd
    else:
        h = np.diag(ho[:,1]-ho[:,0])
        _, g_array = g_r_verlet(p/nd,bins,rc,h/nd)
        return g_array

def gr_MD(pos,bins,rc,h):
    '''RDF computing function

        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        rc : float,
            Cutoff radius for g(r)
        bins : int
            Number of bins
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        Returns
        -------
        g_array: float, dimension (bins,)
            Normalized histograms to be equivalent to the g(r)
    '''
    _, g_array = g_r_verlet(pos,bins,rc,h)
    return g_array

def local_sq(pos, h = None, N=200, wdt=500, cs=3, ms=30, dump = False, correction_grid = None):
    _, array_sq = s_q_from_pos_smear_array(pos, h = h, N=N, wdt=wdt, cs=cs, ms=ms, dump = dump, correction_grid = correction_grid)
    return array_sq

def load_custom_raw(filename, skip_to_np=3, skip_to_box=5, skip_to_pos=9):
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
    
    nparticles = np.genfromtxt(filename, skip_header=skip_to_np,  dtype=np.int,    max_rows=1).item(0)
    box        = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:,:2]
    h=box
    rawdata    = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata=rawdata[np.argsort(rawdata[:,0])]
    pos = np.zeros([nparticles, 3])
    if skip_to_pos == 9:
        pos[:,:] = rawdata[:,2:5]
    if skip_to_pos == 15:
        pos[:,:] = rawdata[:,4:7]

    return pos, h, rawdata

def unwrap_trajectories(u_pos, pos, h):
    '''Load LAMMPS .data type configuration files

        Parameters
        ----------
        u_pos : ndarray of shape (Natoms,3,steps)
            Unwrapped Real coordinates of atoms
        pos : ndarray
            Real coordinates of atoms in the initial step
        h : ndarray
            Simulation box size (c1|c2|c3)
        Returns
        -------
        u_pos : ndarray of shape (Natoms,3,steps)
            Unwrapped Real coordinates of atoms
    '''
    atoms = u_pos.shape[0]
    step = u_pos.shape[2]
    off_vec = np.zeros((atoms, 3))
    for i in range(step - 1):
        pos_new = pos[(i + 1) * atoms:(i + 2) * atoms].copy()
        if i == 0:
            pos_old = pos[(i) * atoms:(i + 1) * atoms].copy()
        new_off = pbc_msd(pos_new, pos_old, h)
        off_vec += new_off
        pos_old = pos_new
        u_pos[:, :, i + 1] = pos_new + off_vec
    return u_pos.copy()

def read_init(filename):
    if 'data' in filename:
        atoms = read(filename, 0, format='lammps-data',style='atomic')
    else:
        atoms = read(filename, 0, format='lammps-dump-text')
    init_pos = atoms.get_positions()
    h = np.zeros((3,2))
    h[:,1] = np.diag(atoms.get_cell())
    return init_pos, h

def get_kpoints(h, n_unit_cells, system, delta_x=0.0005, points=[r'$\Gamma$',r'$X$',r'$W$',r'$K$',r'$\Gamma$',r'$L$']):
    # Define symmetry points for different systems
    if system == 'fcc':
        sym_points = {'$\Gamma$':[0,0,0], '$X$':[0.5,0,0], '$W$':[0.5,0.25,0], '$K$':[0.375,0.375,0], '$L$':[0.25,0.25,0.25]}
    elif system == 'bcc':
        sym_points = {'$\Gamma$':[0,0,0], '$H$':[0,0,0.5], '$P$':[0.25,0.25,0.25], '$N$':[0,0.25,0.25]}
    elif system == 'sc':
        sym_points = {'$\Gamma$':[0,0,0], '$X$':[0,0.25,0], '$M$':[0.25,0.25,0], '$R$':[0.25,0.25,0.25]}
    elif system == 'diamond':
        sym_points = {'$\Gamma$':[0,0,0], '$X$':[0.5,0,0], '$W$':[0.5,0.25,0], '$K$':[0.375,0.375,0], '$L$':[0.25,0.25,0.25]}

    # Extract symmetry points and calculate input k-points
    kpointsIn = np.array([sym_points[point] for point in points])
    nKin = kpointsIn.shape[0]
    
    # Calculate distances between successive points
    distances = np.linalg.norm(np.diff(kpointsIn, axis=0), axis=1)
    
    # Determine number of interpolating points between each pair of symmetry points
    nInterp_per_segment = (distances / delta_x).astype(int)
    
    kpoints = [kpointsIn[0]]
    for i in range(nKin - 1):
        # Generate interpolated points between current and next symmetry point
        # interp_x = np.linspace(0, 1, nInterp_per_segment[i], endpoint=False)
        # segment_points = (1 - interp_x[:, None]) * kpointsIn[i] + interp_x[:, None] * kpointsIn[i + 1]
        segment_points = np.linspace(kpointsIn[i], kpointsIn[i + 1], nInterp_per_segment[i], endpoint=False)
        kpoints.extend(segment_points)
    kpoints.append(kpointsIn[-1])  # Add the final point

    # Convert k-points to reciprocal space
    hin = np.mean(h[:, 1] - h[:, 0])
    kpoints = np.array(kpoints) * 2 * np.pi * n_unit_cells * 2 / hin

    # Calculate the position for the symmetry lines
    nK = len(kpoints)
    lines = np.cumsum([0] + nInterp_per_segment.tolist())

    return np.array(kpoints), lines

def fourier_summation(fractional_hess, kpoints, R, natoms_cell):
    dd = np.einsum('ij,mnj',np.exp(1j*np.dot(kpoints,R.T)),fractional_hess)
    u,v = np.linalg.eig(dd[:,:3*natoms_cell,:3*natoms_cell])
    return np.sort(u,axis=1)

def map_to_ref(init_pos, h, natoms_cell = 1, ref_pos = None):
    if ref_pos is None:
        ref_pos = init_pos[0,:]
    
    box = np.diag(h[:,1]-h[:,0])
    hinv = np.linalg.inv(box)
    s = np.dot(init_pos-ref_pos,hinv)
    init_pos = (np.dot(s-np.round(s),box))
    Rj = np.transpose(init_pos.reshape(-1,natoms_cell,3),[1,0,2]) 
    return Rj.mean(0)

def get_fractional_hess(hess, natoms_cell, nnc, n_unit_cells):
    fractional_hess = np.zeros((natoms_cell*3,natoms_cell*3,nnc*n_unit_cells**3))
    for i in range(nnc*n_unit_cells**3):
        fractional_hess[:,:,i] = hess[i*natoms_cell*3:(i+1)*natoms_cell*3,:natoms_cell*3]

    return fractional_hess