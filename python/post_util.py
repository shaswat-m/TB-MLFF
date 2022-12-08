import sys, os
#sys.path.append(os.path.realpath(os.environ['DPNgit']+'/python'))
from md_util import *
import re

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

