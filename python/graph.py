# --------------------------------------------------------
# graph.py
# by SangHyuk Yoo, shyoo@yonsei.ac.kr
# last modified : Sat Aug 20 12:06:49 KST 2022
#
# Objectives
# Prepare the graphs before training from the LAMMPS data
# 
# Prerequisites library
# 1. ASE(Atomistic Simulation Environment)
# 2. DGL
# 3. PyTorch
#
# Usage 
# python3 graph.py --input-file train_info.json
# --------------------------------------------------------

# import modules
import argparse
import json
import os

import numpy as np

from ase import io, neighborlist, Atoms
import torch
import dgl
from dgl.data import DGLDataset

from tqdm import tqdm

# define class
class MDDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 r_cut=0.0,
                 traj_index=None,
                 snap_index=None):
        """ Template for customizing graph datasets in DGL.

        Parameters
        ----------
        url : str
            URL to download the raw dataset
        raw_dir : str
            Specifying the directory that will store the
            downloaded data or the directory that
            already stores the input data.
            Default: ~/.dgl/
        save_dir : str
            Directory to save the processed dataset.
            Default: the value of `raw_dir`
        force_reload : bool
            Whether to reload the dataset. Default: False
        verbose : bool
            Whether to print out progress information
        r_cut : float
            cut off radius
        traj_index : dict
            trajectory indeices
            start : step : end
        snap_index : dict
            snapshot indeices for each trajectory
            start : step : end
        """
        self.r_cut = r_cut
        self.trajectories = range(traj_index['start'],
                                  traj_index['end'],
                                  traj_index['step'])
        self.snapshots = range(snap_index['start'],
                               snap_index['end'],          
                               snap_index['step'])

        self.graphs = []
        self.labels = []
        self.data_name = os.path.join(save_dir, 'data.graph')

        super(MDDataset, self).__init__(name='MDDataset',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def process(self):
        """
            TODO
            ----
            1. How is the label determined?
        """
        for traj in tqdm(self.trajectories):
            for snap in self.snapshots:
                filename = os.path.join(self.raw_dir, 
                                        f'traj_{traj}', f'dump.lj.{snap}')
                atoms = io.read(filename, 0, format='lammps-dump-text')
                graph = self._generate_graph(atoms)
                self.graphs.append(graph)

                label = traj*10000 + snap
                self.labels.append(label)

        self.labels = torch.Tensor(self.labels).int()
        return self.graphs, self.labels

    def save(self):
        dgl.save_graphs(self.data_name, 
                        self.graphs, 
                        {'labels': self.labels})

    def load(self):
        graphs, label_dict = dgl.load_graphs(self.data_name)
        self.graphs = graphs
        self.labels = label_dict['labels']
        return self.graphs, self.labels

    def has_cache(self):
        if os.path.exists(self.data_name):
            return True
        else:
            return False

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def _generate_graph(self,
                        atoms:Atoms=None) -> dgl.DGLGraph:
        """generate a graph using ASE

        Parameters
        ----------
        atoms: ASE Atoms

        Returns
        -------
        graph: DGLGraph
        
        
        """
        # check the atoms type
        if not isinstance(atoms, Atoms):
            raise TypeError('It is not ASE Atoms type.')
        
        # get src ids, dst ids, ditance and unit vectors
        # using ASE neighbor list construction function
        src_ids, dst_ids, distance, unit_vec \
            = neighborlist.neighbor_list('ijdD', atoms, self.r_cut)

        # build graph
        graph = dgl.graph((src_ids, dst_ids), idtype=torch.int32)
        
        # get forces and save themas node data
        forces = atoms.get_forces()
        forces = torch.from_numpy(forces).float()
        graph.ndata['forces'] = forces

        # save distance and unit vectors as edge data
        graph.edata['distance'] \
            = torch.tensor(distance).reshape((graph.num_edges(), 1)).float()
        distance = np.repeat(distance, 3).reshape((graph.num_edges(), 3))
        graph.edata['unit_vec'] \
            = torch.from_numpy(np.divide(unit_vec, distance)).float()

        return graph

if __name__ == '__main__':
    # define argparse instance
    parser = argparse.ArgumentParser('input parser')
    parser.add_argument('--input-file',
                        required=False,
                        help='input file with full path',
                        type=str,
                        default='dataset_info.json')

    # parse arguments
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f'{args.input_file} file is not found.')
    else:
        with open(args.input_file, 'r') as fh:
            params = json.load(fh)
            data_params = params['data_params']

    # initialize parameters for MD dataset
    url = None
    raw_dir = data_params['raw_dir']
    save_dir = data_params['save_dir']
    force_reload = data_params['force_reload']
    verbose = True

    r_cut = data_params['r_cut']
    traj_index = data_params['traj_index']
    snap_index = data_params['snap_index']

    # create dataset
    dataset = MDDataset(url=url, raw_dir=raw_dir, save_dir=save_dir, 
                        force_reload=force_reload, verbose=verbose,
                        r_cut=r_cut, traj_index=traj_index, snap_index=snap_index)