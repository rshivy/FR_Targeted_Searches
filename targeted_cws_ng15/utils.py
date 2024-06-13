#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:04:17 2024

@author: bjornlarsen
"""

import json, os
import la_forge.core as co
import numpy as np

def get_initial_sample(pta, dataset, outdir_label, project_path,
                       pt_chains=True, try_alt_dir=False):
    """
    Get initial sample as MLV from a previous run

    Parameters
    ----------
    psrname : string, name of pulsar
    model_label : string, name of model from which to take a sample

    Returns
    -------
    Dictionary of param names and sample value
    """
    chaindir = f'{project_path}/data/chains/{dataset}'
    if try_alt_dir:
        chaindir = f'{chaindir}_old_Tspans/{outdir_label}'
    else:
        chaindir = f'{chaindir}/{outdir_label}'
    try:
        with open(chaindir+'/model_params.json' , 'r') as fin:
            model_params = json.load(fin)
    except:
        with open(chaindir+'/0/model_params.json' , 'r') as fin:
            model_params = json.load(fin)
    try:
        corepath = f'{chaindir}/core.h5'
        c = co.Core(corepath=corepath, params=model_params,
                    pt_chains=pt_chains)
    except:
        if pt_chains:
            chain_file = 'chain_1.0.txt'
        else:
            chain_file = 'chain_1.txt'
        if not os.path.isfile(f'{chaindir}/{chain_file}'):
            chaindir += '/1'
        c = co.Core(chaindir=chaindir, params=model_params,
                    pt_chains=pt_chains)
    param_dict = c.get_map_dict()
    x0 = []
    rand_sample_pnames = []
    for i, pname in enumerate(pta.param_names):
        if pname in list(param_dict.keys()):
            x0.append(param_dict[pname])
        else:
            #print(f'no sample for {pname} in {model_label} chain')
            rand_sample_pnames.append(pname)
            x0.append(pta.params[i].sample())
        if 'log10_rho' in pname:
            # pta.params is not the same size as pta.param_names if using FS
            i -= 1
    if len(rand_sample_pnames) > 0:
        print(f'no initial samples for params {rand_sample_pnames}')
    return np.hstack(x0)