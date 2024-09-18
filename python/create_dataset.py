# --------------------------------------------------------
# create_dataset.py
# by SangHyuk Yoo, shyoo@yonsei.ac.kr
#    Shaswat Mahanty, shaswatm@stanford.edu
# last modified : Tue 29 Nov 2022 08:44:19 PM KST
#
# Objectives
# Run MD simulation for liquid Argon using LJ potential model
# 
# Prerequisites library
# 1. ASE(Atomistic Simulation Environment)
# 2. liblammps.so(LAMMPS shared library)
# 3. LAMMPS Python wrapper
#
# Usage 
# python3 create_dataset.py
# --------------------------------------------------------

# import modules
import argparse
import os
import random

import numpy as np

from ase import units, Atom
from lammps import lammps

# initial parameters for lammps
# LJ Argon system - 
#MASS = 39.9 # amu
EPSILON = 0.238 # kcal/mol
SIGMA = 3.4 # Angstrom
CUTOFF = 2.5 # LJ unit
SKIN = 2.0 # Angstrom

# define argparse instance
parser = argparse.ArgumentParser('input parser')
parser.add_argument('--save_dir', 
                    required=False,
                    type=str,
                    default=os.getcwd(),
                    help='save directory')
parser.add_argument('--start_idx_traj',
                    required=False,
                    type=int,
                    help='start index of trajectories',
                    default=0)
parser.add_argument('--num_traj',
                    required=False,
                    type=int,
                    help='number of trajectories to produce', 
                    default=1)
parser.add_argument('--num_atom', 
                    required=False, 
                    type=int,
                    help='number of atoms',
                    default=256)
parser.add_argument('--density',
                    required=False,
                    type=float,
                    help='LJ density of Argon',
                    default=0.844)
parser.add_argument('--temperature_init',
                    required=False,
                    type=float,
                    help='initial temperature for NVT ensemble (K)',
                    default=10)
parser.add_argument('--temperature_final',
                    required=False,
                    type=float,
                    help='final temperature temperature for (K)',
                    default=105)
parser.add_argument('--pressure',
                    required=False,
                    type=float,
                    help='pressure (bar)',
                    default=137.3)
parser.add_argument('--num_vacancy',
                    required=False,
                    type=int,
                    help='Create vacancies',
                    default=0)
parser.add_argument('--potential_file',
                    required=False,
                    type=str,
                    help='Potential file to use',
                    default='lj')
parser.add_argument('--material',
                    required=False,
                    type=str,
                    help='material',
                    default='Ar')
parser.add_argument('--time_step',
                    required=False,
                    type=float,
                    help='time step (ps)',
                    default=0.01078)
parser.add_argument('--run_step',
                    required=False,
                    type=int,
                    help='run step ',
                    default=200000)
parser.add_argument('--dump_freq',
                    required=False,
                    type=int,
                    help='dump frequency ',
                    default=100)

# define class
class RunMD:
    def __init__(self):
        # set Ar LJ parameters for lammps
        # TODO: it will move to setup
        self.mass = MASS  # amu
        self.epsilon = EPSILON * (units.kcal/units.mol)  # eV
        self.sigma = SIGMA  # Angstrom
        self.cutoff = CUTOFF * self.sigma  # Angstrom
        self.skin = SKIN  # Angstrom

        # set indices
        self.idx_proc = 0
        self.idx_traj = 0
   
    def setup(self, args):
        # get trajectory info.
        self.idx_proc = args.pid
        self.idx_traj = args.idx_traj + args.start_idx_traj

        # get save directory
        self.save_dir = args.save_dir
        self.save_dir_md = os.path.join(self.save_dir, f'traj_{self.idx_traj}')
        if not os.path.exists(self.save_dir_md):
            os.mkdir(self.save_dir_md)
        self.filename_log = os.path.join(self.save_dir_md, 'log.traj')
        self.filename_dump = os.path.join(self.save_dir_md, 'dump.lj.*')
        self.filename_dump_one = os.path.join(self.save_dir_md, 'dump.lj')

        # calculate cell size and replication number
        # TODO: cell can be orthogonal.
        self.density = args.density
        self.num_atom = args.num_atom
        self.volume = ((self.num_atom/self.density)*self.sigma**3)
        self.cell = self.volume**(1/3)
        self.num_replicate = (self.num_atom/4)**(1/3)
        self.num_vacancy = args.num_vacancy

        # setup MD simulation parameters
        self.dump_freq = args.dump_freq
        self.log_freq = 200
        self.time_step = args.time_step  # ps
        self.num_step = 200000
        self.run_step = args.run_step
        self.nvt_thermo_freq = 0.02
        self.npt_thermo_freq = 0.02
        self.npt_baro_freq = 0.2
        self.num_chains = 5
        self.num_mtk = 5
        self.temp_obj = args.temperature_init  # K
        self.temp_init = args.temperature_init  # K
        self.temp_final = args.temperature_final  # K
        self.press_obj = args.pressure  # bar
    
        # get random number for initial velocity
        # TODO: Maximum value should be changable
        #np.random.seed(self.idx_proc)
        self.random = np.random.randint(100000)

    def print_setup(self):
        print('=================================')
        print(f'PID: {self.idx_proc}')
        print(f'save directory: {self.save_dir}')
        print(f'trajectory indices: {self.idx_traj}')
        print('=================================')
        print(f'mass: {self.mass} amu')
        print(f'epsilon: {self.epsilon} eV')
        print(f'sigma: {self.sigma} Angstrom')
        print(f'cutoff: {self.cutoff} Angstrom')
        print(f'skin distnace : {self.skin} Angstrom')
        print(f'number of atoms in FCC: {self.num_atom}')
        if self.num_vacancy > 0:
            print(f'number of vacancies : {self.num_vacancy}')
            print(f'number of atoms : : {self.num_atom - self.num_vacancy}')
            self.idx_vac = random.sample(range(0, self.num_atom), self.num_vacancy)
            self.idx_vac = ' '.join(map(str, self.idx_vac))
        print(f'cell length : {self.cell} Angstrom')
        print('=================================')
        print(f'time step : {self.time_step} ps')
        print(f'number of steps : {self.num_step}')
        print(f'random number for velocity initialization : {self.random}')
        print(f'NPT : ')
        print(f'\tTarget Temperature = {self.temp_obj}')
        print(f'\tTarget Pressure = {self.press_obj}')
        print(f'\tThermostat Frequency = {self.npt_thermo_freq} ps')
        print(f'\tBarostat Frequency = {self.npt_baro_freq} ps')
        print(f'NVT : ')
        print(f'\tInit Temperature = {self.temp_init}')
        print(f'\tFinal Temperature = {self.temp_final}')
        print(f'\tThermostat Frequency = {self.nvt_thermo_freq} ps')

    def run(self):
        # create instances
        cmdargs = ['-log', f'{self.filename_log}.{self.idx_traj}', \
                   '-screen', 'none']
        self.lmp = lammps(cmdargs=cmdargs)

        # setup lammps
        self.lmp.command('units metal')
        self.lmp.command('boundary p p p')
        self.lmp.command('atom_style atomic')

        # create lattice
        self.lmp.command(f'lattice fcc {self.cell/self.num_replicate}')
        self.lmp.command(f'region box block 0 {self.num_replicate} \
                                            0 {self.num_replicate} \
                                            0 {self.num_replicate}')
        self.lmp.command(f'create_box 1 box')
        self.lmp.command(f'create_atoms 1 box')

        if self.num_vacancy > 0:
            self.lmp.command(f'group rm_vac id {self.idx_vac}')
            self.lmp.command(f'delete_atoms group rm_vac')
        self.lmp.command(f'mass 1 {self.mass}')
        self.lmp.command(f'velocity all create {self.temp_obj} {self.random} \
                           dist gaussian mom yes rot yes')

        # define force parameters
        if args.potential_file == 'lj':
            self.lmp.command(f'pair_style lj/cut {self.cutoff}')
            self.lmp.command(f'pair_coeff 1 1 {self.epsilon} {self.sigma} {self.cutoff}')

            # define neighbor build scheme
            self.lmp.command(f'neighbor {self.skin} bin')
            self.lmp.command(f'neigh_modify check yes')
        else:
            if '.eam' in args.potential_file:
                self.lmp.command(f'pair_style eam')
                self.lmp.command(f'pair_coeff * * {args.potential_file}')
            else:
                raise NotImplementedError
        # run NPT ensemble
        self.lmp.command(f'thermo {self.log_freq}')
        self.lmp.command(f'timestep {self.time_step}')
        self.lmp.command(f'fix 1 all npt temp {self.temp_obj} {self.temp_obj} {self.npt_thermo_freq} \
                           tchain {self.num_chains} tloop {self.num_mtk} \
                           iso {self.press_obj} {self.press_obj} {self.npt_baro_freq}')
        self.lmp.command(f'run {self.num_step}')
        self.lmp.command(f'unfix 1')
        
        # run NVT ensemble (1)
        self.lmp.command(f'fix 1 all nvt temp {self.temp_obj} {self.temp_obj} {self.nvt_thermo_freq} \
                           tchain {self.num_chains} tloop {self.num_mtk}')
        self.lmp.command(f'run {self.num_step}')

        # define dump conditions
        self.lmp.command(f'dump 1 all custom {self.dump_freq} {self.filename_dump} \
                           id type x y z vx vy vz fx fy fz')
        self.lmp.command(f'dump 2 all custom {self.dump_freq} {self.filename_dump_one} \
                           id type xu yu zu vx vy vz')
        self.lmp.command(f'thermo {self.log_freq}')

        # run NVT ensemble (2) : those configuration will be used in ML.
        self.lmp.command(f'unfix 1')
        self.lmp.command(f'fix 1 all nvt temp {self.temp_init} {self.temp_final} {self.nvt_thermo_freq} \
                           tchain {self.num_chains} tloop {self.num_mtk}')
        self.lmp.command(f'reset_timestep 0')
        self.lmp.command(f'run {self.run_step}')
    
    def quit(self):
        self.lmp.close()

def main(args, i):
    # append new argument: PID, idx_traj
    # TODO : make idx_traj as global variable 
    pid = os.getpid()
    args.pid = pid
    args.idx_traj = i
    
    # create MD instances
    md = RunMD()

    # setup MD instances
    md.setup(args)
    md.print_setup()

    # run MD
    md.run()

    # quit
    md.quit()

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    a1 = Atom(args.material)
    MASS = a1.mass
    for i in range(0, args.num_traj):
        main(args, i)
