# --------------------------------------------------------
# post_process.py
# by Shaswat Mohanty, shaswatm@stanford.edu
# last modified : Sat Oct 29 2022 12:13:23 2022
#
# Objectives
# Post processing analysis in the benchmarking tests
#
#
# Usage 
# python3 post_process.py --input-file post_process.json
# --------------------------------------------------------

import argparse
import json
import os, sys
from copy import deepcopy
import itertools
from functools import partial
import multiprocessing as mp
import numpy as np

from pyscal_util import *
from md_util import *
from post_util import *

class post_analysis():
    def __init__(self, timestep = 0.01078, dump_filename = 'dump_name', func = 'MSD',
                 gnn_md_dir = 'GNN_dir', save_dir = 'save_data', dump_freq = 100,
                 total_timestep = 50000, freq = 100, lj_md = 'lammps'):
        self.timestep      = timestep
        self.dump_filename = dump_filename
        self.func          = func
        self.gnn_md_dir    = gnn_md_dir+'/gnn'
        self.lj_md         = lj_md
        if lj_md == 'ase':
            self.md_dir = gnn_md_dir + '/lj'
        self.save_dir      = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.total_timestep= total_timestep
        self.dump_freq     = dump_freq
        self.steps         = int(total_timestep/dump_freq)
        self.freq          = freq
        self.ins           = self.steps // self.freq
        self.t             = np.linspace(0,self.timestep*self.dump_freq*self.freq*(self.ins-1),self.ins)
        self.nd            = 3.405

    def init_parallel(self):
        if self.func == 'XPCS':
            self.pool = mp.Pool(8)
        elif self.func == 'XSVS':
            self.pool = mp.Pool(8)
        else:
            self.pool = mp.Pool(mp.cpu_count())

    def load_LAMMPS(self):
        if self.lj_md == 'lammps':
            self.md_pos, self.hins, self.N_atoms = load_dumpfile_atom_data_fast(self.dump_filename, self.steps, 1, verbose=False, h_full=False)
            self.hin = self.hins[:, :, 0]
            self.h = np.diag(self.hin[:, 1] - self.hin[:, 0])
        else:
            filename = self.md_dir + '/data.lj.0'
            p, ho = load_custom(filename, 2, 4, 11)
            self.N_atoms = p.shape[0]
            self.md_pos = np.zeros((self.N_atoms*self.steps,3))
            for i in range(self.steps):
                filename = self.md_dir + '/data.lj.%d' % (self.dump_freq * i + self.dump_freq)
                p, ho = load_custom(filename, 2, 4, 11)
                self.md_pos[i * self.N_atoms:(i + 1) * self.N_atoms, :] = p / self.nd

            self.h = np.diag(ho[:, 1] - ho[:, 0]) / self.nd

    def load_GNN(self):
        self.gnn_pos = np.zeros(self.md_pos.shape)
        for i in range(self.steps):
            filename = self.gnn_md_dir + '/data.lj.%d' % (self.dump_freq * i + self.dump_freq)
            p, ho = load_custom(filename, 2, 4, 11)
            self.gnn_pos[i * self.N_atoms:(i + 1) * self.N_atoms, :] = p / self.nd

        self.h_gnn = np.diag(ho[:, 1] - ho[:, 0]) / self.nd

    def cal_rdf_and_sq(self):
        r0 = 0.25  # minimum r for g(r) histogram
        rc = 4.0;  # cut-off radius of g(r) calculation
        bins = np.arange(r0, rc + 0.01, 0.01)
        N = 400
        wdt = 400
        cs = 5
        ms = 40
        scale_pos = np.zeros((3, 3))
        r_array = 0.5 * (bins[:-1] + bins[1:])
        if self.lj_md == 'lammps':
            compute_MD = partial(gr_MD, bins=bins, rc=rc, h=self.h)
            g_arr_MD = self.pool.map(compute_MD,
                                [self.md_pos[i * self.freq * self.N_atoms:(i * self.freq + 1) * self.N_atoms, :].copy() for i in range(self.ins)])
            scale_pos[1, :] = self.hin[:, 1] - self.hin[:, 0]
        else:
            compute_MD = partial(gr_GNN, bins=bins, rc=rc, nd=self.nd)
            g_arr_MD = self.pool.map(compute_MD,
                                      [self.md_dir + '/data.lj.%d' % (self.dump_freq * self.freq * i) for i in
                                       range(self.ins)])
            filename = self.md_dir + '/data.lj.0'
            p, ho = load_custom(filename, 2, 4, 11)
            scale_pos[1, :] = ho[:, 1] - ho[:, 0]
        g_arr_MD = np.array(g_arr_MD)
        gr_md = g_arr_MD.mean(0)
        correct_MD = s_q_from_pos_smear(scale_pos, N=N, wdt=wdt, cs=cs, ms=ms, dump=True, structure_factor=True,
                                        correction=True)
        compute_GNN = partial(gr_GNN, bins=bins, rc=rc, nd=self.nd)
        g_arr_GNN = self.pool.map(compute_GNN, [self.gnn_md_dir + '/data.lj.%d' % (self.dump_freq * self.freq * i) for i in range(self.ins)])
        g_arr_GNN = np.array(g_arr_GNN)
        gr_gnn = g_arr_GNN.mean(0)
        filename = self.gnn_md_dir + '/data.lj.0'
        p, ho = load_custom(filename, 2, 4, 11)
        scale_pos[1, :] = ho[:, 1] - ho[:, 0]
        correct_GNN = s_q_from_pos_smear(scale_pos, N=N, wdt=wdt, cs=cs, ms=ms, dump=True, structure_factor=True,
                                         correction=True)

        np.savetxt(self.save_dir+'/GNN_MD_GR.txt', np.stack([r_array, gr_md, gr_gnn], axis=1))
        plt.xlim(0, 4)
        plt.ylim(0, 3.5)
        plt.plot(r_array, gr_md, 'b:', label='MD')
        plt.plot(r_array, gr_gnn, 'k--', label='GNN')
        plt.xlabel(r'r')
        plt.ylabel(r'g(r)')
        plt.legend()
        plt.savefig(self.save_dir+'/MD_GNN_g_r.png')
        plt.clf()
        an_md_pos = np.zeros((self.N_atoms + 2, 3, self.steps))
        an_gnn_pos = np.zeros((self.N_atoms + 2, 3, self.steps))
        ################## S(q) ####################
        for i in range(self.steps):
            temp_pos = self.md_pos[i * self.N_atoms:(i + 1) * self.N_atoms, :]
            an_md_pos[:2, :, i] = self.hin.T
            an_md_pos[2:, :, i] = temp_pos
            filename = self.gnn_md_dir + '/data.lj.%d' % (self.dump_freq * i)
            ps, hs = gr_GNN(filename, nd=self.nd, database=True)
            an_gnn_pos[:2, :, i] = hs.T
            an_gnn_pos[2:, :, i] = ps

        q_md, _ = s_q_from_pos_smear_array(an_md_pos[:, :, 0], h=None, N=N, wdt=wdt, cs=cs, ms=ms, dump=True,
                                           correction_grid=correct_MD)
        q_gnn, _ = s_q_from_pos_smear_array(an_gnn_pos[:, :, 0], h=None, N=N, wdt=wdt, cs=cs, ms=ms, dump=True,
                                            correction_grid=correct_GNN)

        compute_sq_MD = partial(local_sq, h=None, N=N, wdt=wdt, cs=cs, ms=ms, dump=True, correction_grid=correct_GNN)
        sq_array_MD = self.pool.map(compute_sq_MD, [an_md_pos[:, :, self.freq * i] for i in range(self.ins)])
        sq_array_MD = np.array(sq_array_MD)
        sq_md = sq_array_MD.mean(0)

        compute_sq_GNN = partial(local_sq, h=None, N=N, wdt=wdt, cs=cs, ms=ms, dump=True, correction_grid=correct_GNN)
        sq_array_GNN = self.pool.map(compute_sq_GNN, [an_gnn_pos[:, :, self.freq * i] for i in range(self.ins)])
        sq_array_GNN = np.array(sq_array_GNN)
        sq_gnn = sq_array_GNN.mean(0)

        np.savetxt(self.save_dir+'/MD_SQ.txt', np.stack([q_md[5:], sq_md[5:]], axis=1))
        np.savetxt(self.save_dir+'/GNN_SQ.txt', np.stack([q_gnn[5:], sq_gnn[5:]], axis=1))
        plt.ylim(0, 3.5)
        plt.xlim(0, 3.5)
        plt.plot(q_md[1:], sq_md[1:], 'b:', label='MD')
        plt.plot(q_gnn[1:], sq_gnn[1:], 'k--', label='GNN')
        plt.xlabel(r'q')
        plt.ylabel(r's(q)')
        plt.legend()
        plt.savefig(self.save_dir+'/MD_GNN_s_q.png')
        plt.clf()

    def cal_MSD(self):
        msd_md = np.zeros(self.ins)
        msd_gnn = np.zeros(self.ins)
        u_pos_md = np.zeros((self.N_atoms, 3, self.steps))
        u_pos_gnn = np.zeros((self.N_atoms, 3, self.steps))
        u_pos_md[:, :, 0] = self.md_pos[:self.N_atoms, :].copy()
        u_pos_gnn[:, :, 0] = self.gnn_pos[:self.N_atoms, :].copy()
        u_pos_md = unwrap_trajectories(u_pos_md.copy(), self.md_pos, self.h)
        u_pos_gnn = unwrap_trajectories(u_pos_gnn.copy(), self.gnn_pos, self.h_gnn)
        for i in range(1, self.ins):
            diff_arr_md = u_pos_md[:, :, 0] - u_pos_md[:, :, self.freq * i]
            msd_md[i] = np.mean(np.linalg.norm(diff_arr_md, axis=1) ** 2)
            diff_arr_gnn = u_pos_gnn[:, :, 0] - u_pos_gnn[:, :, self.freq * i]
            msd_gnn[i] = np.mean(np.linalg.norm(diff_arr_gnn, axis=1) ** 2)

        slope_md, _ = curve_fit(objective, self.t, msd_md)
        slope_gnn, _ = curve_fit(objective, self.t, msd_gnn)
        conv_un = self.nd ** 2 * 1e4
        np.savetxt(self.save_dir+'/MSD_values.txt', np.stack([self.t, msd_md, msd_gnn], axis=1))
        plt.plot(self.t, msd_md, 'r', label='MD')
        plt.plot(self.t, objective(self.t, slope_md), 'r--', label=r'D$_MD$=%.2f $\mu$m$^2$/s' % (slope_md * conv_un / 6))
        plt.plot(self.t, msd_gnn, 'b', label='GNN')
        plt.plot(self.t, objective(self.t, slope_gnn), 'b--', label=r'D$_MD$=%.2f $\mu$m$^2$/s' % (slope_gnn * conv_un / 6))
        plt.legend()
        plt.xlabel(r'Time (fs)')
        plt.ylabel(r'MSD (LJ)')
        plt.savefig(self.save_dir+'/MSD_plots.png')
        plt.clf()

    def cal_XPCS(self):
        atoms_add = 45
        tester = XPCS_Suite(filename=self.dump_filename, atoms_add=atoms_add)
        tester.ext = 'GNN'
        tester.timestep = 10
        tester.nondim_t = 10.78e-15
        tester.nondim_d = self.nd * 1e-10
        tester.dir_name = self.save_dir + '/'
        h_pos = np.zeros((atoms_add + 2, 3, self.steps))
        for i in range(1, self.steps + 1):
            filename = self.gnn_md_dir + '/data.lj.%d' % (self.dump_freq * i)
            p, ho = load_custom(filename, 2, 4, 11)
            bl = np.mean(ho[:, 1] - ho[:, 0])
            s = p / bl
            s = s - 0.5
            s = s - np.round(s)
            p = bl * (s + 0.5)
            temp_pos = p[:atoms_add, :]
            h_pos[:2, :, i - 1] = ho.T
            h_pos[2:, :, i - 1] = temp_pos
        tester.box_length = bl / self.nd
        tester.N_atoms = p.shape[0]
        tester.h = ho / self.nd
        tester.hin = ho / self.nd
        tester.pos_input = h_pos.copy() / self.nd
        q_vals = [1.57, 3.14, 4.44, 5.43, 6.28]
        store = []
        for q in q_vals:
            temp = deepcopy(tester)
            temp.pool = self.pool
            temp.q_val = q
            temp.store_gamma_workflow()
            temp.save_correlations()
            store.append([temp.gamma_ns, q ** 2, q])

        store = np.array(store)
        popt, _ = curve_fit(objective_lin, store[:, 1], store[:, 0])
        a, b = popt
        diff = a * temp.nondim_d ** 2 * 1e21
        np.savetxt(self.save_dir+'/Gamma_GNN.txt', store)
        print(r'The diffusivity is %.2f um.m/s' % diff)

        testing = XPCS_Suite(filename=self.dump_filename, atoms_add=atoms_add)
        testing.ext = 'MD'
        testing.nondim_d = self.nd * 1e-10
        testing.dir_name = self.save_dir +'/'
        if self.lj_md == 'lammps':
            testing.load_and_process_trajectory()
        else:
            testing.timestep = 10
            testing.nondim_t = 10.78e-15
            h_pos = np.zeros((atoms_add + 2, 3, self.steps))
            for i in range(1, self.steps + 1):
                filename = self.md_dir + '/data.lj.%d' % (self.dump_freq * i)
                p, ho = load_custom(filename, 2, 4, 11)
                bl = np.mean(ho[:, 1] - ho[:, 0])
                s = p / bl
                s = s - 0.5
                s = s - np.round(s)
                p = bl * (s + 0.5)
                temp_pos = p[:atoms_add, :]
                h_pos[:2, :, i - 1] = ho.T
                h_pos[2:, :, i - 1] = temp_pos
            testing.box_length = bl / self.nd
            testing.N_atoms = p.shape[0]
            testing.h = ho / self.nd
            testing.hin = ho / self.nd
            testing.pos_input = h_pos.copy() / self.nd
        q_vals = [1.57, 3.14, 4.44, 5.43, 6.28]
        store = []
        for q in q_vals:
            temp = deepcopy(testing)
            temp.pool = self.pool
            temp.q_val = q
            temp.store_gamma_workflow()
            temp.save_correlations()
            store.append([temp.gamma_ns, q ** 2, q])

        store = np.array(store)
        popt, _ = curve_fit(objective_lin, store[:, 1], store[:, 0])
        a, b = popt
        diff = a * temp.nondim_d ** 2 * 1e21
        np.savetxt(self.save_dir+'/Gamma_MD.txt', store)
        print(r'The diffusivity is %.2f um.m/s' % diff)

    def cal_XSVS(self):
        tester = XPCS_Suite(filename=self.dump_filename, atoms_add=atoms_add)
        tester.ext = 'GNN'
        tester.timestep = 10
        tester.nondim_t = 10.78e-15
        tester.nondim_d = self.nd * 1e-10
        tester.dir_name = self.save_dir + '/'
        h_pos = np.zeros((atoms_add + 2, 3, self.steps))
        for i in range(1, self.steps + 1):
            filename = self.gnn_md_dir + '/data.lj.%d' % (self.dump_freq * i)
            p, ho = load_custom(filename, 2, 4, 11)
            bl = np.mean(ho[:, 1] - ho[:, 0])
            s = p / bl
            s = s - 0.5
            s = s - np.round(s)
            p = bl * (s + 0.5)
            temp_pos = p[:atoms_add, :]
            h_pos[:2, :, i - 1] = ho.T
            h_pos[2:, :, i - 1] = temp_pos
        tester.box_length = bl / self.nd
        tester.N_atoms = p.shape[0]
        tester.h = ho / self.nd
        tester.hin = ho / self.nd
        tester.pos_input = h_pos.copy() / self.nd
        testing = XPCS_Suite(filename=self.dump_filename, atoms_add=45)
        testing.ext = 'MD'
        testing.nondim_d = self.nd * 1e-10
        testing.dir_name = self.save_dir +'/'
        if self.lj_md == 'lammps':
            testing.load_and_process_trajectory()
        else:
            testing.timestep = 10
            testing.nondim_t = 10.78e-15
            h_pos = np.zeros((atoms_add + 2, 3, self.steps))
            for i in range(1, self.steps + 1):
                filename = self.md_dir + '/data.lj.%d' % (self.dump_freq * i)
                p, ho = load_custom(filename, 2, 4, 11)
                bl = np.mean(ho[:, 1] - ho[:, 0])
                s = p / bl
                s = s - 0.5
                s = s - np.round(s)
                p = bl * (s + 0.5)
                temp_pos = p[:atoms_add, :]
                h_pos[:2, :, i - 1] = ho.T
                h_pos[2:, :, i - 1] = temp_pos
            testing.box_length = bl / self.nd
            testing.N_atoms = p.shape[0]
            testing.h = ho / self.nd
            testing.hin = ho / self.nd
            testing.pos_input = h_pos.copy() / self.nd
        q_vals = [1.57, 3.14, 4.44, 5.43, 6.28]
        for q in q_vals:
            temp_md = deepcopy(testing)
            temp_md.pool = self.pool
            temp_md.q_val = q
            temp_md.get_intensity_workflow()
            temp_md.contrast_computation()
            temp_gnn = deepcopy(tester)
            temp_gnn.pool = self.pool
            temp_gnn.q_val = q
            temp_gnn.get_intensity_workflow()
            temp_gnn.contrast_computation()
            np.savetxt(self.save_dir+'/contrast_at_%.2f.txt'%q, np.stack([temp_md.contrast_data[:,0], temp_md.contrast_data[:,1], temp_gnn.contrast_data[:,1]],axis=1))
            plt.plot(temp_md.contrast_data[:,0], temp_md.contrast_data[:,1], lw =2)
            plt.plot(temp_gnn.contrast_data[:, 0], temp_gnn.contrast_data[:, 1], lw=2)
            plt.xlabel(r'$t$ (ps)')
            plt.ylabel(r'$\beta(q)$')
            plt.savefig(self.save_dir + '/contrast_at_%.2f.png'%q)
            plt.clf()

    def cal_interface(self):
        gnn_part = partial(fraction_liquid_ase, natoms=self.N_atoms)
        lf_gnn = self.pool.map(gnn_part, [self.gnn_md_dir + '/data.lj.%d' % (i + 1) for i in range(self.steps)])
        if self.lj_md == 'lammps':
            md_part = partial(fraction_liquid_lammps, hi=self.hins[:, :, 0], natoms=self.N_atoms)
            lf_md = self.pool.map(md_part, [self.md_pos[i * self.N_atoms:(i + 1) * self.N_atoms, :] for i in range(self.steps)])
        else:
            md_part = partial(fraction_liquid_ase, natoms=self.N_atoms)
            lf_md = self.pool.map(md_part, [self.md_dir + '/data.lj.%d' % (i + 1) for i in range(self.steps)])
        lf_gnn = np.array(lf_gnn)
        lf_md = np.array(lf_md)
        np.savetxt(self.save_dir+'/LF_values.txt', np.stack([self.t, lf_md, lf_gnn], axis=1))
        plt.plot(self.t, lf_md, 'r', label='MD')
        plt.plot(self.t, lf_gnn, 'b', label='GNN')
        plt.legend()
        plt.xlabel(r'Time (ps)')
        plt.ylabel(r'Liquid fraction')
        plt.savefig(self.save_dir+'/LF_plots.png')
        plt.clf()

    def cal_NN_dist(self):
        hinv_md = np.linalg.inv(self.h)
        hinv_gnn = np.linalg.inv(self.h_gnn)
        d_md = np.zeros(self.steps)
        d_gnn = np.zeros(self.steps)
        d2_md = np.zeros(self.steps)
        d2_gnn = np.zeros(self.steps)
        fatom = 109

        for i in range(self.steps):
            faid = i * self.N_atoms + fatom
            r_md = self.md_pos[faid, :] - self.md_pos[faid + 1, :]
            r2_md = self.md_pos[faid, :] - self.md_pos[faid + 16, :]
            r_gnn = self.gnn_pos[faid, :] - self.gnn_pos[faid + 1, :]
            r2_gnn = self.gnn_pos[faid, :] - self.gnn_pos[faid + 16, :]
            d_md[i] = np.linalg.norm(pbc(r_md, self.h, hinv_md))
            d_gnn[i] = np.linalg.norm(pbc(r_gnn, self.h_gnn, hinv_gnn))
            d2_md[i] = np.linalg.norm(pbc(r2_md, self.h, hinv_md))
            d2_gnn[i] = np.linalg.norm(pbc(r2_gnn, self.h_gnn, hinv_gnn))

        np.savetxt(self.save_dir+'/Hist_values2.txt', np.stack([self.t[:len(d_md)], d_md, d_gnn, d2_md, d2_gnn], axis=1))
        plt.hist(d_md, 100, density=True, alpha=0.5, label='MD')
        plt.hist(d_gnn, 100, density=True, alpha=0.5, label='GNN')
        plt.legend()
        plt.ylabel(r'Time (ps)')
        plt.xlabel(r'Count')
        plt.savefig(self.save_dir+'/Hist_plots2.png')
        plt.clf()


def main(inputs):
    post_params = inputs['post_params']


    os.makedirs(post_params['save_dir'], exist_ok=True)
    PA = post_analysis(timestep=post_params['timestep'], dump_filename=post_params['dump_filename'],
                       func=post_params['func'], gnn_md_dir=post_params['gnn_md_dir'],
                       save_dir=post_params['save_dir'], total_timestep=post_params['total_timestep'],
                       dump_freq=post_params['dump_freq'], freq=post_params['freq'],
                       lj_md=post_params['lj_md'])
    if PA.func != 'XPCS':
        PA.load_LAMMPS()
    if PA.func == 'rdf_and_sq':
        PA.init_parallel()
        PA.cal_rdf_and_sq()
    elif PA.func == 'XPCS':
        PA.init_parallel()
        PA.cal_XPCS()
    elif PA.func == 'XSVS':
        PA.init_parallel()
        PA.cal_XSVS()
    elif PA.func == 'MSD':
        PA.load_GNN()
        PA.cal_MSD()
    elif PA.func == 'interface':
        PA.load_GNN()
        PA.init_parallel()
        PA.cal_interface()
    elif PA.func == 'NN_dist':
        PA.load_GNN()
        PA.cal_NN_dist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='input file', type=str, default='post_process.json')
    args = parser.parse_args()

    with open(args.input_file, 'r') as fh:
        input_params = json.load(fh)

    main(input_params)