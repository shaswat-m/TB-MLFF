## TB-MLFF: A testbed for benchmarking machine-learning force fields
This library comprises the end-to-end workflow of developing a machine-learned force field using graph neural networks, to evaluating the molecular dynamics trajectory run by using a series of benchmarking tests. 

----
## Cite
We would appreciate if you could cite the following works depending on what components of this library you use:

Please cite the following paper for using the testbed library to benchmark your force-field:
* Evaluating the Transferability of Machine-Learned Force Fields for Material Property Modeling [https://doi.org/10.1016/j.cpc.2023.108723]

If you used the Graph Neural Network to train the model the please cite:
* Z. Li et al., J. Chem. Phys. (2022), 156, 144103 [https://doi.org/10.1063/5.0083060]

If you use the XPCS analysis then please cite:
* Computational Approaches to Model X-ray Photon Correlation Spectroscopy from Molecular Dynamics [https://doi.org/10.1088/1361-651X/ac860c]

If you used the liquid fraction analysis please cite:
* pyscal: A python module for structural analysis of atomic environments [https://doi.org/10.21105/joss.01824]

----
# Installation
## 1. Set up python enable LAMMPS
Install lammps and set up liblammps.so required for running LAMMPS using the python wrapper.

```commandline
mkdir codes ;
cd codes ;
codedir=$(pwd) ;
git clone -b release https://github.com/lammps/lammps.git mylammps ;
cd mylammps/src ;
make yes-manybody ;
make yes-mc ;
make yes-rigid ;
make yes-molecule ;
make yes-kspace ;
make mode=shared mpi ;
```
## 2. Install the TB-MLFF dependencies
Download the benchmarking repository and set up the conda virtual environment required for creating the dataset and training the machine-learned force field.
```commandline
cd $codedir ;
git clone git@gitlab.com:micronano_public/tb-mlff.git ; 
cd tb-mlff ;
workdir=$(pwd) ;
conda create --name ml_train --file spec-file-train.txt ;
conda activate ml_train ;
conda install -c anaconda gcc ;
cd $codedir/mylammps/src ;
make install-python ;
conda deactivate ;
```
Note that to carry out the post-processing analysis pyscal may be required which can be installed using the following. In addition, the GNN MD will be run using a cpu enabled pytorch and dgl so we create a separate conda environement.
```commandline
cd $workdir ;
conda create --name ml_bench --file spec-file-bench.txt ;
conda activate ml_bench ;
conda install -c conda-forge pybind11 ;
cd $codedir
git clone https://github.com/pyscal/pyscal.git ;
cd pyscal ;
python setup.py install ;
conda deactivate ;
```
----
# Usage
## 1. How to prepare dataset
Prepare the dataset using the following: The default parameters used in the paper have been provided.
```commandline
conda activate ml_train ;
python3 $workdir/python/create_dataset.py --save_dir $workdir/data --num_traj 10;  
python3 $workdir/python/graph.py --input-file $workdir/configs/train_info.json ;
```

## 2. How to train GNN
On preparing the dataset and creating the subsequent graphs, we can train the graph neural network using the default parameters used in the paper.
```commandline
python3 $workdir/python/train.py --input-file $workdir/configs/train_info.json ;
conda deactivate ;
```
Alternatively you can download our pre-trained model
```commandline
cd $workdir/runs/trial_run_1 ;
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1waYb1oEap0DyOBI6ezZBwxb7R2XMOZY7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1waYb1oEap0DyOBI6ezZBwxb7R2XMOZY7" -O best.ckpt && rm -rf /tmp/cookies.txt ;
```
## 3. How to run MD
We are currently running GNN-MD by using the OpenMM and ASE. The assumptions is that the LJ interatomic potential is run using LAMMPS and dumps to a single dump file. However, we can also use this script to run alternate cases  in `test_info.json`:
* We can use the LJ interatomic potential on openMM to compute the MD trajectory by switching from `"ff_model":"gnn"` to `"ff_model":"lj"`.
* We can plot the parity plot by using `"test_mode":"ms"` and `"test_name":"parity"`.
```commandline
conda activate ml_bench ;
python3 $workdir/python/test.py --input-file $workdir/configs/train_info.json $workdir/configs/test_info.json ;
```

## 4. Post processing results
Download the LAMMPS trajectory of 5000 frames for a 4000 atom system run for 538 ps at 95 K
```commandline
cd $workdir/dumpfiles ;
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1F2qjnNp8pb3qoy_mEOXcRJj6tSn3RhIN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F2qjnNp8pb3qoy_mEOXcRJj6tSn3RhIN" -O dump.ljtest_MD_GAMD_95 && rm -rf /tmp/cookies.txt ;
```
The MD trajectory can be post-processed in a number of ways by making changes  in `post_process.json`: 
* Mean-squared displacement(`"func": "MSD"`) 
* Structure factor and radial distribution function (`"func": "rdf_and_sq"`)
* Computational XPCS (`"func": "XPCS"`) 
* Computational XSVS (`"func": "XSVS"`) 
* Solid-liquid interface analysis (`"func": "interface"`). 
* Incase, you're not using a single dumpfile from lammps but an openMM trajectory by using the LJ interatomic potentional then switch `"lj_md":"lammps"` to `"lj_md":"ase"`.
```commandline
python3 $workdir/python/post_process.py --input-file $workdir/configs/post_process.json ;
```
# Support and Development

For any support regarding the implementation of the source code, contact the developers at: 
* Shaswat Mohanty (shaswatm@stanford.edu)
* SangHyuk Yoo (shyoo08@yonsei.kr.ac.kr)
* Wei Cai (caiwei@stanford.edu)


# Contributing
The development is actively ongoing and the sole contributors are Shaswat Mohanty, SangHyuk Yoo, and Wei Cai.  Request or suggestions for implementing additional functionalities to the library can be made to the developers directly.

# Project status
Development on the neural network model and the post-processing analysis is currently ongoing.


