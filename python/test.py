# --------------------------------------------------------
# test.py
# by SangHyuk Yoo, shyoo@yonsei.ac.kr
#    Shaswat Mohanty, shaswatm@stanford.edu
# last modified : Mon Nov 21 04:34:00 PST 2022
#
# Objectives
# Run MD simulation with NVT ensemble to obtain snapshots by
# Graph Neural Network based force predictor and LJ model.
# In addition, you can choose MS simulation to test 
# Hessian matrix computation and invariant for rotation
# 
# Prerequisites library
# 1. ASE(Atomistic Simulation Environment)
# 2. DGL
# 3. PyTorch
# 4. scikit-learn
# 5. PyTorch-ligthning
# 6. OpenMM with hacked NH integrator (nvt_integrator.py)
#
# Usage 
# python3 train.py --input-file train_info.json test_info.json
# --------------------------------------------------------

# import modules
import argparse
import datetime
import json
import os

import numpy as np

import openmm as mm
from openmm import app, unit
import torch
import pytorch_lightning as pl
import dgl
from ase import Atoms, io, neighborlist
import ase.units as aseUnits
from tqdm import tqdm

from nvt_integrator import Hack1stHalfNoseHooverIntegrator, \
                           Hack2ndHalfNoseHooverIntegrator
from train import LightModel, LightDataModule

# initial parameters for MD
ATOM_TYPE = 'Ar'
MASS = 39.9 # amu
EPSILON = 0.238 # kcal/mol
SIGMA = 3.4 # Angstrom
CUTOFF = 2.5 # LJ unit

# define class
class Test:
    def __init__(self) -> None:
        self.r_cut = 0.0
    
    def generate_graph(self,
                       atoms:Atoms=None) -> dgl.DGLGraph:
        '''Generate a graph using ASE

        Parameters
        ----------
        atoms: ASE Atoms

        Returns
        -------
        graph: DGLGraph
        
        
        '''
        # check the atoms type
        if not isinstance(atoms, Atoms):
            raise TypeError('It is not ASE Atoms type.')
        
        # get src ids, dst ids, ditance and unit vectors
        # using ASE neighbor list construction function
        src_ids, dst_ids, distance, unit_vec \
            = neighborlist.neighbor_list('ijdD', atoms, self.r_cut)

        # build graph
        graph = dgl.graph((src_ids, dst_ids), idtype=torch.int32)

        # save distance and unit vectors as edge data
        graph.edata['distance'] \
            = torch.tensor(distance).reshape((graph.num_edges(), 1)).float()
        distance = np.repeat(distance, 3).reshape((graph.num_edges(), 3))
        graph.edata['unit_vec'] \
            = torch.from_numpy(np.divide(unit_vec, distance)).float()

        return graph

    def init_gnn_model(self,
                       filename:str = None,
                       train_params:dict = None,
                       model_params:dict = None) -> pl.LightningModule:
        model = LightModel(train_params=train_params, 
                           model_params=model_params)
        model = model.load_from_checkpoint(filename, 
                                           train_params=train_params,
                                           model_params=model_params)
        model.eval()

        return model    


class TestMD(Test):
    def __init__(self) -> None:
        super().__init__()
        self.test_mode = 'md'
        self.test_name = None
        self.save_dir = None
    
    def setup(self, args):
        '''setup arguments

        Parameters
        ----------
        args: arguments
        
        '''
        print(datetime.datetime.now())
        self.model_params = args['model_params']
        self.train_params = args['train_params']
        self.data_params  = args['data_params']
        self.test_params  = args['test_params']

        self.ff_model = self.test_params['ff_model']
        if self.ff_model == 'lj':
            self.save_dir = self.train_params['save_dir'] + '/lj'
        elif self.ff_model == 'gnn':
            self.save_dir = self.train_params['save_dir'] + '/gnn'
        else:
            raise ValueError('Wrong input arguments of ff_model in test_info.json')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        os.system("cp %s %s ;"%(os.path.join(self.test_params['save_dir'],self.test_params['dump_filename']),
                                os.path.join(self.save_dir,self.test_params['dump_filename'])))
        os.system("cp %s %s ;"%(os.path.join(self.test_params['save_dir'],'best.ckpt'),
                                os.path.join(self.save_dir,'best.ckpt')))
        self.dump_filename = os.path.join(self.save_dir,
                                          self.test_params['dump_filename'])
        self.log_filename  = os.path.join(self.save_dir,
                                          self.test_params['log_filename'])
    
    def run_lj(self):
        # read initial configurations : lammps dump format
        self.atoms = io.read(self.dump_filename, 0, 'lammps-dump-text')
        num_atoms = len(self.atoms)
        pos  = self.atoms.get_positions()
        cell = self.atoms.get_cell()[:]
        vel  = self.atoms.get_velocities() * aseUnits.fs * 1000 # Angstrom/ps
        atom_type = ATOM_TYPE
        atom_mass = MASS

        # setup Argon system
        topology = app.Topology()
        element  = app.Element.getBySymbol(atom_type)
        chain    = topology.addChain()
        for _ in range(num_atoms):
            residue = topology.addResidue(atom_type, chain)
            topology.addAtom(atom_type, element, residue)
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(cell[0], cell[1], cell[2])

        # create non-bonded force field(LJ)
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(CUTOFF*SIGMA*unit.angstrom)
        nb.setUseDispersionCorrection(False)
        nb.setUseSwitchingFunction(False)

        # add Ar atoms into system and force field dinstance
        for _ in range(num_atoms):
            system.addParticle(atom_mass*unit.amu)
            nb.addParticle(0, SIGMA*unit.angstrom, EPSILON*unit.kilocalorie_per_mole)

        # create center of masss motion remover : it is treated as a force field
        rm = mm.CMMotionRemover()

        # add force field instances into the system
        system.addForce(nb)
        system.addForce(rm)

        # define settings
        timestep     = self.test_params['timestep'] * unit.picoseconds
        temperature  = self.test_params['temperature'] * unit.kelvin
        chain_length = self.test_params['nvt_chain']
        thermo_freq  = self.test_params['thermo_freq'] / unit.picosecond

        # setup integrator
        integrator = mm.NoseHooverIntegrator(temperature, 
                                             thermo_freq,
                                             timestep,
                                             chain_length,
                                             5)

        # define simulator
        device = 'CPU'
        if self.test_params['device'] == 'gpu':
            device = 'CUDA'
        platform = mm.Platform.getPlatformByName(device)
        simulator = app.Simulation(topology=topology,
                                   system=system, 
                                   integrator=integrator,
                                   platform=platform)

        # setup initial condition
        simulator.context.setPositions(pos*unit.angstrom)
        simulator.context.setVelocities(vel*unit.angstrom/unit.picosecond)

        # define logger for OpenMM
        log_freq       = self.test_params['log_freq']
        dump_freq      = self.test_params['dump_freq']
        total_timestep = self.test_params['total_timestep']
        temp_array     = np.linspace(self.test_params['temperature'], 
                                     self.test_params['temperature_end'],
                                     total_timestep)
        dataReporter = app.StateDataReporter(file=self.log_filename, 
                                             reportInterval=log_freq,
                                             totalSteps=total_timestep,
                                             step=True,
                                             kineticEnergy=True,
                                             temperature=True, 
                                             elapsedTime=True,
                                             separator=' ')
        simulator.reporters.append(dataReporter)

        # run OpenMM
        for t in range(1, total_timestep+1):
            simulator.step(1)

            # get new poisitions
            state = simulator.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

            # apply PBC on the position manually
            # OpenMM PBC system does not work correctly
            scaled_pos = np.matmul(pos, np.linalg.inv(cell))
            scaled_pos -= 0.5
            scaled_pos -= np.round(scaled_pos)
            scaled_pos += 0.5
            pos = np.matmul(scaled_pos, cell)
            self.atoms.set_positions(pos)

            # get new velocities
            state = simulator.context.getState(getVelocities=True)
            vel = state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom/unit.picosecond)
            self.atoms.set_velocities(vel / (aseUnits.fs * 1000))

            # update temperature of thermostat
            if t < total_timestep:
                integrator.setTemperature(temp_array[t] * unit.kelvin)		

            # dump files
            if t%dump_freq == 0:
                print(f'Finished {t} steps')
                io.write(filename=os.path.join(self.save_dir, f'data.lj.{t}'),
                         images=self.atoms,
                         format='lammps-data',
                         velocities=True)
        
    def run_gnn(self):
        # read initial configurations : lammps dump format
        self.atoms = io.read(self.dump_filename, 0, 'lammps-dump-text')
        num_atoms = len(self.atoms)
        pos  = self.atoms.get_positions()
        cell = self.atoms.get_cell()[:]
        vel  = self.atoms.get_velocities() * aseUnits.fs * 1000 # Angstrom/ps
        atom_type = ATOM_TYPE
        atom_mass = MASS

        # setup Argon system
        topology = app.Topology()
        element  = app.Element.getBySymbol(atom_type)
        chain    = topology.addChain()
        for _ in range(num_atoms):
            residue = topology.addResidue(atom_type, chain)
            topology.addAtom(atom_type, element, residue)
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(cell[0], cell[1], cell[2])

        # create non-bonded force field like LJ
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(3.0*SIGMA*unit.angstrom)
        nb.setUseDispersionCorrection(False)
        nb.setUseSwitchingFunction(False)

        # define GNN model and parameters
        filename_ckpt = os.path.join(self.save_dir, 'best.ckpt')
        model = self.init_gnn_model(filename_ckpt,
                                    self.train_params,
                                    self.model_params)
        avg_force = model.avg_force.numpy()
        std_force = model.std_force.numpy()

        # add Ar atoms into system and force field dinstance
        for _ in range(num_atoms):
            system.addParticle(atom_mass*unit.amu)
            nb.addParticle(0, SIGMA*unit.angstrom, EPSILON*unit.kilocalorie_per_mole,)

        # create center of masss motion remover : it is treated as a force field
        rm = mm.CMMotionRemover()

        # add force field instances into the system
        system.addForce(nb)
        system.addForce(rm)

        # define settings
        timestep     = self.test_params['timestep'] * unit.picoseconds
        temperature  = self.test_params['temperature'] * unit.kelvin
        chain_length = self.test_params['nvt_chain']
        thermo_freq  = self.test_params['thermo_freq'] / unit.picosecond

        # setup integrator using only force 
        dummy_integrator = mm.CompoundIntegrator()
        integrator1 = Hack1stHalfNoseHooverIntegrator(system=system, 
                                                      temperature=temperature,
                                                      collision_frequency=thermo_freq,
                                                      chain_length=chain_length,
                                                      timestep=timestep)
        integrator2 = Hack2ndHalfNoseHooverIntegrator(system=system,
                                                      temperature=temperature,
                                                      collision_frequency=thermo_freq,
                                                      chain_length=chain_length,
                                                      timestep=timestep)
        dummy_integrator.addIntegrator(integrator1)
        dummy_integrator.addIntegrator(integrator2)

        # define simulator
        device = 'CPU'
        if self.test_params['device'] == 'gpu':
            device = 'CUDA'
        platform = mm.Platform.getPlatformByName(device)
        dummy_simulator = app.Simulation(topology=topology,
                                         system=system, 
                                         integrator=dummy_integrator,
                                         platform=platform)

        # setup initial condition
        dummy_simulator.context.setPositions(pos*unit.angstrom)
        dummy_simulator.context.setVelocities(vel*unit.angstrom/unit.picosecond)

        # define logger for OpenMM
        log_freq       = self.test_params['log_freq']
        dump_freq      = self.test_params['dump_freq']
        total_timestep = self.test_params['total_timestep']
        temp_array = np.linspace(self.test_params['temperature'],self.test_params['temperature_end'],total_timestep)
        dataReporter = app.StateDataReporter(file=self.log_filename, 
                                             reportInterval=log_freq,
                                             totalSteps=total_timestep,
                                             step=True,
                                             kineticEnergy=True,
                                             temperature=True, 
                                             elapsedTime=True,
                                             separator=' ')
        dummy_simulator.reporters.append(dataReporter)

        # run MD
        force_unit = 1/(aseUnits.kJ/aseUnits.mol/aseUnits.nm)* \
                    (unit.kilojoule_per_mole/unit.nanometer)
        self.r_cut = self.data_params['r_cut']
        graph = self.generate_graph(atoms=self.atoms)
        batch = [graph, 0]
        with torch.no_grad():
            force = model.forward(batch).numpy()
            force = (force*std_force) + avg_force
            force = force*force_unit # change unit eV/Angstrom to (kJ/mol)/nm

            # save initial configurations
            self.atoms.set_velocities(vel / (aseUnits.fs * 1000))
            io.write(filename=os.path.join(self.save_dir, f'data.lj.0'),
                     images=self.atoms,
                     format='lammps-data',
                     velocities=True)

            for t in range(1, total_timestep+1):
                # 1st NHC
                dummy_integrator.setCurrentIntegrator(0)
                if t != 0:
                    integrator1.copy_state_from_integrator(integrator2)
                integrator1.setPerDofVariableByName('force_last', force)
                dummy_simulator.step(1)

                # get new poisitions
                dummy_state = dummy_simulator.context.getState(getPositions=True)
                pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

                # apply PBC on the position manually
                # OpenMM PBC system does not work correctly
                scaled_pos = np.matmul(pos, np.linalg.inv(cell))
                scaled_pos -= 0.5
                scaled_pos -= np.round(scaled_pos)
                scaled_pos += 0.5
                pos = np.matmul(scaled_pos, cell)
                self.atoms.set_positions(pos)
                
                # calculate new force
                graph = self.generate_graph(atoms=self.atoms)
                batch = [graph, t]
                force = model.forward(batch).numpy()
                force = (force*std_force) + avg_force
                force = force*force_unit 

                # 2nd NHC
                dummy_integrator.setCurrentIntegrator(1)
                integrator2.copy_state_from_integrator(integrator1)
                integrator2.setPerDofVariableByName('gnn_force', force)
                dummy_simulator.step(1)

                # get new velocity
                dummy_state = dummy_simulator.context.getState(getVelocities=True)
                vel = dummy_state.getVelocities(asNumpy=True).value_in_unit(unit.angstrom/unit.picosecond)
                self.atoms.set_velocities(vel / (aseUnits.fs * 1000))
                if t < total_timestep:
                    integrator1.setTemperature(temp_array[t] * unit.kelvin)		
                    integrator2.setTemperature(temp_array[t] * unit.kelvin)		

                # dump files
                if t%dump_freq == 0:
                    print(f'Finished {t} steps')
                    io.write(filename=os.path.join(self.save_dir, f'data.lj.{t}'),
                             images=self.atoms,
                             format='lammps-data',
                             velocities=True)
    def run(self):
        if self.ff_model == 'lj':
            self.run_lj()
        elif self.ff_model == 'gnn':
            self.run_gnn()
        else:
            raise ValueError(f'lj or gnn only acceptable for force field: \
                               {self.ff_model}') 

    def quit(self):
        print(datetime.datetime.now())

class TestMS(Test):
    def __init__(self) -> None:
        super().__init__()
        self.test_mode = 'ms'
        self.test_name = None
        self.save_dir  = None
        self.k_mat_lmp = None
        self.k_mat_gnn = None
        self.atoms     = None

    def setup(self, args: dict=None) -> None:
        '''setup arguments

        Parameters
        ----------
        args: dict
            arguments for Tests
        
        Returns
        -------
        None

        '''
        self.model_params = args['model_params']
        self.train_params = args['train_params']
        self.data_params  = args['data_params']
        self.test_params  = args['test_params']

        self.test_name = self.test_params['test_name']
        self.save_dir  = self.train_params['save_dir']
        self.dump_filename = os.path.join(self.save_dir,
                                          self.test_params['dump_filename'])

        self.r_cut = self.data_params['r_cut']
        if self.test_params['ff_model'] == 'gnn':
            self.filename_ckpt = os.path.join(self.save_dir, 'best.ckpt')
        else:
            raise ValueError('ff_model in TestMS should be gnn')

        if self.test_name == 'hessian':
            self.hess_lmp_filename = os.path.join(self.save_dir, 'dynmat.dat') 
            self.hess_filename = os.path.join(self.save_dir, 'hessian.npz')

    def test_hessian(self,
                     dx:float = 1e-6) -> None:
        '''Calculate Hessian matrix 
           and test it is comparable with LAMMPS results

        Parameters
        ----------
        dx: float
            distance to move atoms (LJ unit)

        Returns
        -------
        None

        TODO
        ----
        1. Add OpenMM or LAMMPS methods to compute Hessian matrix
        
        '''
        # read hessian matrix for lammps
        k_mat_lmp = np.loadtxt(self.hess_lmp_filename)
        k_dof = int((k_mat_lmp.shape[0]*k_mat_lmp.shape[1])**0.5)
        self.k_mat_lmp = k_mat_lmp.reshape(k_dof, k_dof)
        self.k_mat_gnn = np.zeros((k_dof, k_dof))

        # read initial configuration : lammps dump format
        self.atoms = io.read(self.dump_filename, 0, 'lammps-dump-text')
        pos = self.atoms.get_positions()

        # initialize gnn model
        model = self.init_gnn_model(self.filename_ckpt,
                                    self.train_params,
                                    self.model_params)     
        avg_force = model.avg_force.numpy()
        std_force = model.std_force.numpy()

        # calculate hessian matrix
        k_unit = SIGMA**2/(EPSILON*(aseUnits.kcal/aseUnits.mol))
        dx = SIGMA*dx
        with torch.no_grad():
            for i in tqdm(range(k_dof)):
                # move forward and evaluate
                pos_fwd = pos.copy()
                pos_fwd[i//3, i%3] += dx
                self.atoms.set_positions(pos_fwd)
                batch = [self.generate_graph(self.atoms), 0]
                force_fwd = model.forward(batch).numpy()
                force_fwd = (force_fwd * std_force) + avg_force

                # move backward and evaluate
                pos_bwd = pos.copy()
                pos_bwd[i//3, i%3] -= dx
                self.atoms.set_positions(pos_bwd)
                batch = [self.generate_graph(self.atoms), 0]
                force_bwd = model.forward(batch).numpy()
                force_bwd = (force_bwd * std_force) + avg_force

                # calculate row vector of hessian matrix
                k_mat_gnn_row = (force_bwd - force_fwd)/(2*dx)*k_unit
                self.k_mat_gnn[i, :] = k_mat_gnn_row.reshape(-1,)

        # save hessian matrix
        np.savez(self.hess_filename,
                 hess_gnn=self.k_mat_gnn,
                 hess_lammps=self.k_mat_lmp)

        # check the matrix is symmetric
        if np.allclose(self.k_mat_lmp, self.k_mat_lmp.T):
            print('LAMMPS Hessian is symmateric.')
        else:
            print('LAMMPS Hessian is not symmateric.')
            
        if np.allclose(self.k_mat_gnn, self.k_mat_gnn.T):
            print('GNN-MD Hessian is symmateric.')
        else:
            print('GNN-MD Hessian is not symmateric.')

    def test_invariant(self) -> None:
        '''Test invariant for translation and rotation

        Parameters
        ----------
        None

        Returns
        -------
        None

        TODO
        ----
        1. Make a function to save results
        '''

        # read initial configuration : lammps dump format
        self.atoms = io.read(self.dump_filename, 0, 'lammps-dump-text')
        num_atoms = len(self.atoms)
        pos = self.atoms.get_positions()
        
        # initialize gnn model
        model = self.init_gnn_model(self.filename_ckpt,
                                    self.train_params,
                                    self.model_params)     
        avg_force = model.avg_force.numpy()
        std_force = model.std_force.numpy()

        # make rotation matrix by random variable
        ang_x, ang_y, ang_z = 2*np.pi*np.random.rand(3)
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(ang_x), np.sin(ang_x)],
                       [0.0,-np.sin(ang_x), np.cos(ang_x)]])
        ry = np.array([[ np.cos(ang_y), 0.0, np.sin(ang_y)],
                       [0.0, 1.0, 0.0], 
                       [-np.sin(ang_y), 0.0, np.cos(ang_y)]])
        rz = np.array([[ np.cos(ang_z), np.sin(ang_z), 0.0], 
                       [-np.sin(ang_z), np.cos(ang_z), 0.0], 
                       [0.0, 0.0, 1.0]])
        rmat = rx @ ry @ rz

        # update positions : translation and rotation
        pos_trans = pos.copy() + \
                    np.repeat(1e1*np.random.rand(3).reshape(1,3), 
                              num_atoms,
                              axis = 0)
        pos_rot = (rmat @ (pos.copy()).T).T

        # calculate forces
        with torch.no_grad():
            # forces without any transformations
            batch = [self.generate_graph(self.atoms), 0]
            force = model.forward(batch).numpy()
            force = (force * std_force) + avg_force

            # transform forces
            f_trans_ref = force.copy()
            f_rot_ref   = (rmat @ (force.copy()).T).T

            # forces with translational positions
            self.atoms.set_positions(pos_trans)
            batch = [self.generate_graph(self.atoms), 0]
            f_trans_pred = model.forward(batch).numpy()
            f_trans_pred = (f_trans_pred * std_force) + avg_force

            self.atoms.set_positions(pos_rot)
            batch = [self.generate_graph(self.atoms), 0]
            f_rot_pred = model.forward(batch).numpy()
            f_rot_pred = (f_rot_pred * std_force) + avg_force

        if np.allclose(f_trans_ref,f_trans_pred):
            print('The force field is translationally invariant')
        else:
            print('The force field is not translationally invariant')
        if np.allclose(f_rot_ref,f_rot_pred):
            print('The force field is rotationally invariant')
        else:
            print('The force field is not rotationally invariant')

    def test_parity(self):
        dm = LightDataModule(train_params=self.train_params,
                             data_params=self.data_params)
        dm.setup()
        val_dataloader = dm.val_dataloader()
        model = self.init_gnn_model(self.filename_ckpt,
                                    self.train_params,
                                    self.model_params)
        avg_force = model.avg_force.numpy()
        std_force = model.std_force.numpy()
        forces_pred = []
        forces_refs = []
        with torch.no_grad():
            count = 0
            for batch in val_dataloader:
                pred = model.forward(batch).detach().numpy()
                pred = (pred * std_force) + avg_force
                refs = batch[0].ndata["forces"].detach().numpy()

                forces_pred.append(pred)
                forces_refs.append(refs)

                if count % 100 == 0:
                    print('update = {}'.format(count))

                count += 1

        plt.figure(figsize=[4, 4], dpi=300, constrained_layout=True)
        plt.xlabel(r'$\mathbf{F}_{pred} (eV/\AA)$', fontsize=12)
        plt.ylabel(r'$\mathbf{F}_{ref} (eV/\AA)$', fontsize=12)
        plt.scatter(np.concatenate(forces_pred).reshape(-1), np.concatenate(forces_refs).reshape(-1), color='k', s=3)
        plt.plot(np.linspace(-5, 5), np.linspace(-5, 5), '--r')
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

        filename_plot = self.save_dir + '/preds_vs_refs.png'
        plt.savefig(filename_plot)
        plt.clf()

    def run(self):
        if self.test_name == 'hessian':
            self.test_hessian()
        elif self.test_name == 'invariant':
            self.test_invariant()
        elif self.test_name == 'parity':
            self.test_parity()
        else:
            raise ValueError(f'hessian, parity or invariant only acceptable: \
                               {self.test_name}')

    def quit(self):
        pass

def main(args):
    # check arguments
    test_mode = args['test_params']['test_mode']
    if test_mode == 'md':
        test = TestMD()
    else:
        test = TestMS() 

    # setup instances
    test.setup(args)

    # run
    test.run()

    # quit
    test.quit()

if __name__ == '__main__':
    # define argparse instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file',
                        help='two input files ',
                        type=str,
                        nargs='*',
                        default='train_info.json, test_info.json')

    # parse arguments
    args = parser.parse_args()
    train_info = args.input_file[0]
    test_info  = args.input_file[1]
    if not os.path.exists(train_info):
        raise FileNotFoundError(f'{train_info} file is not found.')
    else:
        with open(train_info, 'r') as fh:
            train_params = json.load(fh)

    if not os.path.exists(test_info):
        raise FileNotFoundError(f'{test_info} file is not found.')
    else:
        with open(test_info, 'r') as fh:
            test_params = json.load(fh)

    input_params = {**train_params, **test_params}

    # run main
    main(input_params)
