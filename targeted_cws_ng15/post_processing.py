#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:22:17 2024

@author: bjornlarsen
"""

from matplotlib import rcParams
rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 300
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import json, os, glob, logging, copy, shutil
logger = logging.getLogger(__name__)
import corner
import astropy.units as u
import la_forge.core as co
import la_forge.diagnostics as dg
from la_forge.utils import bayes_fac
from la_forge.rednoise import gorilla_bf
from enterprise.signals import parameter
import enterprise.constants as const
from QuickCW.PulsarDistPriors import PXDistPrior, DMDistPrior
from ipta_gwb_analysis import diagnostics
from targeted_cws_ng15.empirical_distr_new import (
                                    make_empirical_distributions_from_core)
from targeted_cws_ng15.convolve_priors import convolved_prior
from DR3_noise_modeling.custom_plotting import plot_rednoise_spectrum

def get_prior_distr(core, param, Npoints=1000):
    idx = core.params.index(param)
    pline = core.priors[idx]
    prior_type = pline[pline.index(':')+1:pline.index('(')]
    # setup x-axis for the plot
    if prior_type == 'Uniform':
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        x = np.linspace(pmin, pmax, 300)
        y = parameter.UniformPrior(x, pmin, pmax)
        #prior_dist = sps.uniform(loc=pmin, scale=pmax-pmin)
        #y = prior_dist.pdf(x)
    elif prior_type == 'Normal':
        mu = float(pline[pline.index('mu')+3:pline.index(', sigma')])
        sigma = float(pline[pline.index('sigma')+6:-1])
        #prior_dist = sps.norm(loc=mu, scale=sigma)
        pmin = np.min([mu-3*sigma, np.min(core.chain[core.burn:,idx])])
        pmax = np.max([mu+3*sigma, np.max(core.chain[core.burn:,idx])])
        x = np.linspace(pmin, pmax, 300)
        y = parameter.NormalPrior(x, mu, sigma)
        #y = prior_dist.pdf(x)
    elif prior_type == 'LinearExp':
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        x = np.linspace(pmin, pmax, 300)
        y = parameter.LinearExpPrior(x, pmin, pmax)
    elif prior_type == 'PXDist':
        dist = float(pline[pline.index('dist=')+5:pline.index(', err')])
        err = float(pline[pline.index('err')+4:-1])
        pmin = np.min([dist-3*err, np.min(core.chain[core.burn:,idx])])
        pmax = np.max([dist+3*err, np.max(core.chain[core.burn:,idx])])
        x = np.linspace(pmin, pmax, 300)
        y = PXDistPrior(x, dist, err)
    elif prior_type == 'DMDist':
        dist = float(pline[pline.index('dist=')+5:pline.index(', err')])
        err = float(pline[pline.index('err')+4:-1])
        pmin = np.min([dist-2*err, np.min(core.chain[core.burn:,idx])])
        pmax = np.max([dist+2*err, np.max(core.chain[core.burn:,idx])])
        x = np.linspace(pmin, pmax, 300)
        y = DMDistPrior(x, dist, err)
    elif prior_type == 'Convolve':
        x, y = convolved_prior(core, param, Npoints=Npoints)
    else:
        raise ValueError(f'No distribution defined yet for {prior_type}')
    return x, y

def get_bayes_fac(c, amp_param='log10_mc'):
    idx = c.params.index(amp_param)
    pmin = float(c.priors[idx][c.priors[idx].index('pmin')+5:
                               c.priors[idx].index(', pmax')])
    pmax = float(c.priors[idx][c.priors[idx].index('pmax')+5:
                               c.priors[idx].index(')')])
    BF, _ = bayes_fac(c.chain[c.burn:, idx], logAmin=pmin, logAmax=pmax)
    if np.isnan(BF):
        BF = gorilla_bf(c.chain[c.burn:, idx], max=pmax, min=pmin, nbins=20)
    return BF

def load_chains(chaindir, source_name, dataset, outdir_path, cs=None):
    # load chaindirs
    if not os.path.isdir(chaindir):
        raise ValueError('Chaindir does not exist~!!!')
    chaindirs = [x[0]+'/' for x in os.walk(chaindir)][1:]
    nchains = len(chaindirs)
    
    # load model params
    try:
        with open(chaindir+'model_params.json' , 'r') as fin:
            model_params = json.load(fin)
    except:
        try:
            with open(chaindirs[0]+'model_params.json' , 'r') as fin:
                model_params = json.load(fin)
        except:
            with open(chaindirs[3]+'model_params.json' , 'r') as fin:
                model_params = json.load(fin)
    stats = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
    
    # load cores
    cs = []
    for i, cd in enumerate(chaindirs):
        print(f'{i+1}/{nchains}')
        try:
            core = co.Core(label=source_name, chaindir=cd,
                           params=model_params+stats, pt_chains=False)
            if len(core.chain) > 200:
                cs.append(core)
            else:
                print(f'not enough lines in {cd}')
        except:
            print(f'could not load from {cd}')
    return cs, model_params+stats
    
def cs_plot_and_set_burn(cs, source_name='', burn=8000):
    total_samples = 0
    total_postburn_samples = 0
    nchains = len(cs)
    fig, ax = plt.subplots(2,1,figsize=(12,6),sharex=False)
    for i in range(nchains):
        cs[i].set_burn(burn)
        n_samples = len(cs[i].get_param('lnpost',to_burn=True))
        x = np.arange(np.ceil(total_postburn_samples),
                      np.ceil(total_postburn_samples)+
                      np.ceil(n_samples))
        y = cs[i].get_param('lnpost',to_burn=True)
        ax[0].plot(y, alpha=0.2, lw=0.5, c='k')
        if n_samples == 0:
            print(f'no samples for c[{i}]')
        total_postburn_samples += n_samples
        total_samples += len(cs[i].get_param('lnpost',to_burn=False))
        ax[1].plot(x,y,lw=0.5)
    ax[0].set_ylabel(r'$\log$post')
    ax[0].set_ylabel(r'$\log$post')
    ax[0].set_title(source_name)
    ax[1].set_ylabel(r'$\log$post')
    ax[-1].set_xlabel('sample')
    fig.tight_layout()
    #fig.savefig(f'{save_loc}/GR.png', bbox_inches='tight')
    
    print(total_samples,'total samples')
    print(total_postburn_samples,f'samples after burn in (burn = {burn})')
    print(np.round((total_samples-total_postburn_samples)*100/total_samples,3),
          '% of samples burned')
    print(f'Should thin by {get_thin_by(total_postburn_samples)}')
    return cs, get_thin_by(total_postburn_samples)
    
def get_thin_by(n_samples):
    return int(np.max([np.round(n_samples/200000), 1]))

def concatenate_chains_to_single_core(cs, source_name, thin_by, chaindir,
                                      detection=False, upper_limit=False, 
                                      vary_fgw=False, return_core=True,
                                      converged=True):
    chains = []
    for c in cs:
        chains.append(c.chain[c.burn::thin_by])
    chain_array = np.concatenate(chains)
    c = co.Core(chain=chain_array, label=source_name,
                params=cs[0].params, pt_chains=False, burn=0)
    
    # add priors, runtime info
    prior_path = glob.glob(chaindir + '/*/priors.txt')[0]
    c.priors = np.loadtxt(prior_path, dtype=str, delimiter='\t')
    info_path = glob.glob(chaindir + '/*/runtime_info.txt')[0]
    c.runtime_info = np.loadtxt(info_path, dtype=str, delimiter='\t')
    
    # append param name
    c.params = c.params[:-4] + ['log10_h0'] + c.params[-4:]
    if upper_limit:
        # append linear chirp mass
        c.params = c.params[:-4] + ['mc'] + c.params[-4:]
    
    # get log10_fgw, log10_dl
    if vary_fgw:
        log10_fgw = c('log10_fgw')
        pline = [p for p in c.priors if 'log10_fgw' in p][0]
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        log10_fgw_prior = np.array([pmin, pmax])
        # scale for log10_h0 prior calculation
        log10_fgw_prior_scaled = 2/3*log10_fgw_prior
    else:
        line = [c.runtime_info[i] for i in range(len(c.runtime_info))
                if 'log10_fgw' in c.runtime_info[i]][0]
        log10_fgw = float(line.replace('log10_fgw:Constant=',''))
    line = [c.runtime_info[i] for i in range(len(c.runtime_info))
            if 'log10_dL' in c.runtime_info[i]][0]
    log10_dL = float(line.replace('log10_dL:Constant=',''))
    log10_dL_scaled = log10_dL + np.log10(u.Mpc.to(u.m)/const.c)
    
    # append h0 chain and prior
    log10_mc = c('log10_mc',to_burn=False)
    log10_mc_scaled = log10_mc + np.log10(u.Msun.to(u.kg)*const.G/const.c**3)
    # calculate strain
    log10_h0 = (5*log10_mc_scaled/3 + 2*log10_fgw/3 - log10_dL_scaled +
                np.log10(2*np.pi))
    c.chain = np.vstack([c.chain[:,:-4].T,log10_h0,c.chain[:,-4:].T]).T
    #c.chain = np.vstack([c.chain.T,log10_h0]).T
    if upper_limit:
        # append linear chirp mass
        c.chain = np.vstack([c.chain[:,:-4].T,10**log10_mc,c.chain[:,-4:].T]).T
    
    # append h0 prior
    pline = [p for p in c.priors if 'log10_mc' in p][0]
    if upper_limit or detection:
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        log10_mc_prior = np.array([pmin, pmax])
        log10_mc_prior_scaled = (log10_mc_prior +
                                 np.log10(u.Msun.to(u.kg)*const.G/const.c**3))
        if vary_fgw:
            # Now include dL and pi pieces in the scaled log10_mc prior
            # for fgw calulation
            log10_mc_prior_scaled = (5/3*log10_mc_prior_scaled -
                                     log10_dL_scaled + np.log10(2*np.pi))
            # strain prior will be a convolution
            if upper_limit:
                log10_h0_prior_str = ('log10_h0:Convolve(LinearExp(pmin='
                                      f'{log10_mc_prior_scaled[0]}, '
                                      f'pmax={log10_mc_prior_scaled[1]}), '
                                      'Uniform(pmin='
                                      f'{log10_fgw_prior_scaled[0]}, '
                                      f'pmax={log10_fgw_prior_scaled[1]}))')
            else:
                log10_h0_prior_str = ('log10_h0:Convolve(Uniform(pmin='
                                      f'{log10_mc_prior_scaled[0]}, '
                                      f'pmax={log10_mc_prior_scaled[1]}), '
                                      'Uniform(pmin='
                                      f'{log10_fgw_prior_scaled[0]}, '
                                      f'pmax={log10_fgw_prior_scaled[1]}))')
        else:
            # calculate strain prior directly
            log10_h0_prior = (5*log10_mc_prior_scaled/3 + 2*log10_fgw/3 -
                              log10_dL_scaled + np.log10(2*np.pi))
            if upper_limit:
                log10_h0_prior_str=('log10_h0:LinearExp(pmin='
                                    f'{log10_h0_prior[0]}'+
                                    f', pmax={log10_h0_prior[1]})')
            else:
                log10_h0_prior_str = (f'log10_h0:Uniform(pmin='
                                      f'{log10_h0_prior[0]}'+
                                      f', pmax={log10_h0_prior[1]})')
    else:
        log10_mc_mu = float(pline[pline.index('mu')+3:pline.index(', sigma')])
        log10_mc_sigma = float(pline[pline.index('sigma')+6:-1])
        log10_h0_mu = (5*(log10_mc_mu +
                          np.log10(u.Msun.to(u.kg)*const.G/const.c**3))/3 +
                       2*log10_fgw/3 - log10_dL_scaled + np.log10(2*np.pi))
        # hacky way to get h0 sigma
        # just rescale mc sigma by comparing widths of the samples
        r = ((np.max(c.chain[:,c.params.index('log10_h0')]) -
              np.min(c.chain[:,c.params.index('log10_h0')]))/
             (np.max(c.chain[:,c.params.index('log10_mc')]) -
              np.min(c.chain[:,c.params.index('log10_mc')])))
        log10_h0_sigma = r*log10_mc_sigma
        log10_h0_prior_str = (f'log10_h0:Normal(mu={log10_h0_mu}, '+
                              f'sigma={log10_h0_sigma})')
    c.priors = np.concatenate([c.priors, [log10_h0_prior_str]])
    if upper_limit:
        # append linear chirp mass
        log10_mc_prior_str = (f'mc:Uniform(pmin={10**log10_mc_prior[0]}, '+
                              f'pmax={10**log10_mc_prior[1]})')
        c.priors = np.concatenate([c.priors, [log10_mc_prior_str]])
        
    c.save(f'{chaindir}/core.h5')
    
    if converged:
        try:
            with open(chaindir+"/converged.txt", 'w') as f:
                f.write(f'Num chains = {len(cs)}\n')
                f.write(f'burned from each chain = {c.burn}\n')
                f.write(f'thinned by = {thin_by}\n')
                f.write(f'Final num samples = {len(c.get_param("lnpost"))}\n')
        except:
            with open(chaindir+"/converged.txt", 'w') as f:
                f.write(f'Final num samples = {len(c.get_param("lnpost"))}\n')
    if return_core:
        return c
    
def save_higher_temp_cores(cs, c, source_name, thin_by, chaindir,
                           vary_fgw=False, upper_limit=False,
                           return_cores=True):
    if isinstance(cs[0].hot_chains, dict):
        cT = {}
        burn = cs[0].burn
        for T in cs[0].hot_chains.keys():
            print(T)
            chains = [c_i.hot_chains[T][burn::thin_by] for c_i in cs]
            chain_array = np.concatenate(chains)
            cT[T] = co.Core(chain=chain_array, label=source_name,
                            params=c.params, pt_chains=False, burn=0)
            cT[T].runtime_info = c.runtime_info
            cT[T].priors = c.priors
            # get log10_fgw, log10_dl
            if vary_fgw:
                log10_fgw = cT[T]('log10_fgw')
            else:
                line = [c.runtime_info[i] for i in range(len(c.runtime_info))
                        if 'log10_fgw' in c.runtime_info[i]][0]
                log10_fgw = float(line.replace('log10_fgw:Constant=',''))
            line = [c.runtime_info[i] for i in range(len(c.runtime_info))
                    if 'log10_dL' in c.runtime_info[i]][0]
            log10_dL = float(line.replace('log10_dL:Constant=',''))
            log10_dL_scaled = log10_dL + np.log10(u.Mpc.to(u.m)/const.c)
            # append h0 chain and prior
            log10_mc = cT[T]('log10_mc',to_burn=False)
            log10_mc_scaled = log10_mc + np.log10(u.Msun.to(u.kg)*
                                                  const.G/const.c**3)
            # calculate strain
            log10_h0 = (5*log10_mc_scaled/3 + 2*log10_fgw/3 - log10_dL_scaled +
                        np.log10(2*np.pi))
            cT[T].chain = np.vstack([cT[T].chain[:,:-4].T,
                                     log10_h0,cT[T].chain[:,-4:].T]).T
            #c.chain = np.vstack([c.chain.T,log10_h0]).T
            if upper_limit:
                # append linear chirp mass
                cT[T].chain = np.vstack([cT[T].chain[:,:-4].T,
                                         10**log10_mc,cT[T].chain[:,-4:].T]).T
            cT[T].save(f'{chaindir}/core_{T}.h5')
        if return_cores:
            return cT
    else:
        print('Missing hot chains')
        
def plot_hot_chains(c, cT, save_loc, pars=None):
    if pars == None:
        pars = ['cos_inc', 'log10_mc', 'phase0', 
        'lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
    if 'log10_fgw' in c.params:
        pars = ['log10_fgw'] + pars
    if 'crn_gamma' in c.params:
        pars = ['crn_gamma', 'crn_log10_A'] + pars
    dg.plot_chains([c] + list(cT.values()), pars=pars, hist=True,
                   legend_labels=['T=1']+[f'T={T}' for T in list(cT)],
                   ncols=4, title_y=1.10, save=save_loc+"/temp_inspection")
    
def full_chain_diagnostics(c, save_loc, vary_fgw=False, all_sky=False,
                           extra_thinning=True):
    
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(c.get_param('lnpost',to_burn=True), c='k', alpha=1, lw=0.5)
    ax.set_ylabel(r'$\log$post')
    ax.set_xlabel('sample')
    fig.tight_layout()
    fig.savefig(f'{save_loc}/lnpost_trace.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(c.get_param('log10_mc',to_burn=True), c='k', alpha=1, lw=0.2)
    ax.set_ylabel(r'$\log_{10}\mathcal{M}_c$ trace')
    ax.set_xlabel('sample')
    fig.tight_layout()
    fig.savefig(f'{save_loc}/mc_trace.png', bbox_inches='tight')
    
    if vary_fgw:
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(c.get_param('log10_fgw',to_burn=True), c='k', alpha=1, lw=0.2)
        ax.set_ylabel(r'$\log_{10}\mathcal{f}_{\rm{GW}}$ trace')
        ax.set_xlabel('sample')
        fig.tight_layout()
        fig.savefig(f'{save_loc}/log10_fgw_trace.png', bbox_inches='tight')
    
    fig = diagnostics.plot_neff(c, return_fig=True)
    fig.savefig(f'{save_loc}/neff.png', bbox_inches='tight')
    
    fig = diagnostics.plot_grubin(c, M=2, return_fig=True)
    fig.savefig(f'{save_loc}/GR.png', bbox_inches='tight')
    
    if extra_thinning:
        thin_extra = len(c.chain)//5000
    else:
        thin_extra = 1
    idxs = [idx for idx in diagnostics.grubin(c, M=2)[1] if
            idx < len(c.params)-3]
    if vary_fgw:
        idxs += [c.params.index('log10_fgw')]
    if all_sky:
        idxs += [c.params.index('cos_gwtheta'), c.params.index('gwphi')]
    idxs += [c.params.index('cos_inc'), c.params.index('log10_mc'),
             c.params.index('phase0'), c.params.index('psi'),
             c.params.index('lnpost')]
    fig, ax = plt.subplots(len(idxs),1,figsize=(10,np.min([len(idxs),30])),
                           sharex=True)
    for i, idx in enumerate(idxs):
        ax[i].plot(c.get_param(c.params[idx],to_burn=True)[::thin_extra],
                   c='k', alpha=1, lw=0.2)
        ax[i].set_ylabel(c.params[idx],fontsize='xx-small')
    ax[-1].set_xlabel('sample')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(f'{save_loc}/param_traces.png', bbox_inches='tight')
    
def get_mc_prior_mean(core, path, source_name):
    idx = core.params.index('log10_mc')
    pline = core.priors[idx]
    if 'Normal' in pline:
        return float(pline[pline.index('mu')+3:pline.index(', sigma')])
    else:
        prior_path = f'{path}/priors/{source_name}_priors.json'
        with open(prior_path, 'r') as f:
            astro_priors = json.load(f)
        return astro_priors['log10_Mc']
    
def corner_single(target_str, c, dataset, project_path, save_loc=None,
                  vary_crn=False, vary_fgw=False, all_sky=False, 
                  detection=False, upper_limit=False, nogwb=False, pub_fig=False,
                  target_fancy_name=None, truths={}, plot_mc_prior_mean=True,
                  Npoints=1000, pars_select=None, labels=None, titles=None):
    '''
    Parameters
    ----------
    target_str : str
        Name of target
    c : la_forge core
    dataset : str
        Name of dataset
    project_path : str
        Path where you are working
    save_loc : str, optional
        Where to save plot. The default is None.
    vary_crn : bool, optional
        True to indicate CRN is a varied param. The default is False.
    nogwb : bool, optional
        True to indicate CRN/GWB not included in model. The default is False.
    vary_fgw : bool, optional
        True to indicate log10_fgw is a varied param. The default is False.
    all_sky : bool, optional
        True to we are sampling over sky location. The default is False.
    detection : bool, optional
        True to indicate log-uniform chirp mass priors. The default is False.
    upper_limit : bool, optional
        True to indicate uniform chirp mass priors. The default is False.
    pub_fig : bool, optional
        True to remove captions. The default is False.
    target_fancy_name : str, optional
        Alt name of your target for plotting. The default is None.
    truths : dict, optional
        Dictionary of injected values, overplotted. The default is None.
    plot_mc_prior_mean : bool, optional
        True to plot the expected EM chirp mass. The default is True.
    Npoints : int, optional
        Used in integration for convolved priors. The default is 1000.

    Returns
    -------
    None.

    '''
    nsamples = len(c('lnpost'))
    max_Rhat = np.max(diagnostics.grubin(c, M=2)[0])
    
    if target_fancy_name:
        source_name = target_fancy_name
    else:
        source_name = target_str
    if detection:
        save_str = target_str + '_det'
        BF = get_bayes_fac(c)
        print(f'{source_name} SD BF = {np.round(BF, decimals=3)}')
    elif upper_limit:
        save_str = target_str + '_UL'
        UL = c.get_param_credint('mc', onesided=True, interval=95)
        print(f'{source_name} Mc UL = {np.round(UL/1e9, decimals=3)}e9')
    else:
        save_str = target_str + '_astro'
        
    if pub_fig:
        quantiles = None
    else:
        quantiles = [0.16,0.5,0.84]
        for line in c.runtime_info:
            if 'crn_log10_A' in line and not vary_crn and not nogwb:
                crn_log10_A = np.round(float(line[line.index('=')+1:]),
                                       decimals=2)
            if 'crn_gamma' in line and not vary_crn and not nogwb:
                crn_gamma = np.round(float(line[line.index('=')+1:]),
                                     decimals=2)
            if 'log10_fgw' in line and not vary_fgw:
                log10_fgw = np.round(float(line[line.index('=')+1:]),
                                     decimals=2)
            if 'log10_dL' in line:
                log10_dL = np.round(float(line[line.index('=')+1:]),
                                    decimals=2)
            if 'cos_gwtheta' in line and not all_sky:
                cos_gwtheta = np.round(float(line[line.index('=')+1:]),
                                       decimals=2)
            if 'gwphi' in line and not all_sky:
                gwphi = np.round(float(line[line.index('=')+1:]), decimals=2)
        suptitle = (f'{source_name} w/ enterprise \n'
                    f'Dataset: {dataset} \n'
                    r'$\log_{10}d_{\rm{L}} =$' +
                    f'{np.round(log10_dL, decimals=3)} [Mpc] \n' +
                    f'{nsamples} samples \n' +
                    r'Highest $\hat{R}-1$ statistic: ' +
                    f'{np.round(max_Rhat-1,decimals=3)} \n')
        if vary_fgw:
            prior = [p for p in c.priors if 'log10_fgw' in p][0]
            prior = prior.replace('log10_fgw:', r'$\log_{10}f_{\rm{GW}}$: ')
            suptitle += f'{prior} \n'
        else:
            suptitle += r'$\log_{10}f_{\rm{GW}} =$' + f'{log10_fgw} [Hz] \n'
        if all_sky:
            prior = [p for p in c.priors if 'cos_gwtheta' in p][0]
            prior = prior.replace('cos_gwtheta:', r'$\cos\theta$: ')
            suptitle += f'{prior} \n'
            prior = [p for p in c.priors if 'gwphi' in p][0]
            prior = prior.replace('gwphi:', r'$\phi$: ')
            suptitle += f'{prior} \n'
        else:
            suptitle += (r'$\cos\theta =$' + f'{cos_gwtheta} \n'
                         r'$\phi =$' + f'{gwphi} \n')
        if vary_crn:
            prior = [p for p in c.priors if 'crn_log10_A' in p or
                     'gwb_log10_A' in p][0]
            prior = prior.replace('crn_log10_A:', r'$\log_{10}A_{\rm{CRN}}$: ')
            prior = prior.replace('gwb_log10_A:', r'$\log_{10}A_{\rm{GWB}}$: ')
            suptitle += f'{prior} \n'
            prior = [p for p in c.priors if 'crn_gamma' in p or
                     'gwb_gamma' in p][0]
            prior = prior.replace('crn_gamma:', r'$\gamma_{\rm{CRN}}$: ')
            prior = prior.replace('gwb_gamma:', r'$\gamma_{\rm{GWB}}$: ')
            suptitle += f'{prior} \n'
        elif not nogwb:
            suptitle += (r'$\log_{10}A_{\rm{CRN}} =$' + f'{crn_log10_A} \n'
                         r'$\gamma_{\rm{CRN}} =$' + f'{crn_gamma} \n')
        prior = [p for p in c.priors if 'log10_mc' in p][0]
        prior = prior.replace('log10_mc:', r'$\log_{10}\mathcal{M}_c$: ')
        suptitle += f'{prior} \n'
        if detection:
            suptitle += 'SD BF: $\mathcal{B} = $'+f'{np.round(BF, decimals=3)}'
        elif upper_limit:
            suptitle+=(r'$\mathcal{M}_c$ UL ='+f'{np.round(UL/1e9,decimals=3)}'
                       +r'$\cdot 10^9$ M$_\odot$')
        else:
            suptitle += 'informative mass priors'
            
    if pars_select is None and labels is None and titles is None:
        pars_select = ['cos_inc','phase0','psi', 'log10_h0', 'log10_mc']
        #units = ['rad', 'rad', 'rad', 'calculated', r'$M_\odot$']
        titles = [r'$\cos\iota$', '$\Phi_0$', r'$\psi$', r'$\log_{10}h_0$',
                  r'$\log_{10}\mathcal{M}_c$']
        labels = [r'$\cos\iota$', '$\Phi_0$', r'$\psi$', r'$\log_{10}h_0$',
                  r'$\log_{10}\mathcal{M}_c$ [M$_\odot$]']
        if upper_limit:
            pars_select = pars_select[:-1] + ['mc']
            titles = titles[:-1] + [r'$\mathcal{M}_c$']
            labels = labels[:-1] + [r'$\mathcal{M}_c$ [M$_\odot$]']
        if vary_fgw:
            pars_select = ['log10_fgw'] + pars_select
            titles = [r'$\log_{10}f_{\rm{GW}}$'] + titles
            labels = [r'$\log_{10}f_{\rm{GW}}$ [Hz]'] + labels
        if all_sky:
            pars_select = ['cos_gwtheta', 'gwphi'] + pars_select
            titles = [r'$\cos\theta$', r'$\phi$'] + titles
            labels = [r'$\cos\theta$', r'$\phi$'] + labels
        if vary_crn:
            pars_select = ['crn_gamma', 'crn_log10_A'] + pars_select
            #units = ['', ''] + units
            titles = ([r'$\gamma_{\rm{CRN}}$', r'$\log_{10}A_{\rm{CRN}}$'] +
                      titles)
            labels = ([r'$\gamma_{\rm{CRN}}$', r'$\log_{10}A_{\rm{CRN}}$'] +
                      labels)
    else:
        save_str += '_custom_params'

    # plot
    idxs = [c.params.index(p) for p in pars_select]
    fig = corner.corner(c.chain[c.burn:, idxs], labels=labels,
                        quantiles=quantiles, title_quantiles=quantiles,
                        hist_kwargs={'density':True}, rasterized=True,
                        titles=titles, show_titles=True, levels=(0.68, 0.95),
                        label_kwargs={'fontsize': 20},
                        title_kwargs={'fontsize': 14})
    if not pub_fig:
        fig.suptitle(suptitle, fontsize=22, x=1, horizontalalignment='right')

    # Extract the axes
    axes = np.array(fig.axes).reshape((len(idxs), len(idxs)))

    # Loop over the diagonal
    for i, p in enumerate(pars_select):
        ax = axes[i, i]
        # plot prior
        x, y = get_prior_distr(c, p, Npoints=Npoints)
        ax.plot(x, y, 'C2', rasterized=True)
        if not p == 'mc':
            ax.set_xlim([x.min(),x.max()])
        if p == 'log10_mc':
            # add expected chirp mass
            if plot_mc_prior_mean:
                ax.axvline(get_mc_prior_mean(c, project_path, target_str),
                           color='C1', rasterized=True)
        if p == 'mc':
            # add upper limit
            ax.axvline(UL, color='C0')
    # Loop over the histograms
    for yi, p1 in enumerate(pars_select): # rows
        for xi, p2 in enumerate(pars_select[:yi]): # cols
            ax = axes[yi, xi]
            y, _ = get_prior_distr(c, p1)
            x, _ = get_prior_distr(c, p2)
            ax.set_xlim([x.min(),x.max()])
            if not p1 == 'mc':
                ax.set_ylim([y.min(),y.max()])
    # plot true values (if any)
    for true_param in truths:
        if true_param in pars_select:
            idx = pars_select.index(true_param)
            # plot on diagonal
            axes[idx,idx].axvline(truths[true_param], color='r')
            # plot on 2D slices - left
            for i in range(idx):
                axes[idx,i].axhline(truths[true_param], color='r')
            for i in range(idx+1,len(pars_select)):
                axes[i,idx].axvline(truths[true_param], color='r')
    if save_loc:
        if pub_fig:
            fig.savefig(f'{save_loc}/{save_str}.pdf',
                        format='pdf', dpi=600)
        else:
            fig.savefig(f'{save_loc}/{save_str}.png')
            
def corner_single_masked(target_str, c, dataset, project_path, masking_param,
                         mask_range, save_loc=None, vary_crn=False,
                         vary_fgw=False, detection=False, upper_limit=False,
                         pub_fig=False, target_fancy_name=None, truths=None,
                         plot_mc_prior_mean=True, Npoints=1000):
    # make core with masked chain
    c_mask = copy.deepcopy(c)
    mask = ((c_mask(masking_param) > mask_range[0])*
            (c_mask(masking_param) < mask_range[1]))
    chain_new = np.zeros((np.count_nonzero(mask),c_mask.chain.shape[1]))
    for i in range(c_mask.chain.shape[1]):
        chain_new[:,i] = c_mask.chain[:,i][mask]
    c_mask.chain = chain_new
    
    # update prior for masked param
    pline = [p for p in c_mask.priors if 'log10_fgw' in p][0]
    new_prior_str = (f'{masking_param}:Uniform(pmin={mask_range[0]}, '+
                     f'pmax={mask_range[1]})')
    prior_idx = [i for i in range(len(c.priors)) if
                 'log10_fgw' in c_mask.priors[i]][0]
    c_mask.priors[prior_idx] = new_prior_str
    
    # if masking in log10_fgw, need to update log10_h0 prior
    if (upper_limit or detection) and masking_param == 'log10_fgw':
        # log10_dL
        line = [c_mask.runtime_info[i] for i in range(len(c.runtime_info))
                if 'log10_dL' in c.runtime_info[i]][0]
        log10_dL = float(line.replace('log10_dL:Constant=',''))
        log10_dL_scaled = log10_dL + np.log10(u.Mpc.to(u.m)/const.c)
        
        # log10_mc
        pline = [p for p in c_mask.priors if 'log10_mc' in p][0]
        pmin = float(pline[pline.index('pmin')+5:pline.index(', pmax')])
        pmax = float(pline[pline.index('pmax')+5:-1])
        log10_mc_prior = np.array([pmin, pmax])
        log10_mc_prior_scaled = (log10_mc_prior +
                                 np.log10(u.Msun.to(u.kg)*
                                          const.G/const.c**3))
        # Now include dL and pi pieces in the scaled log10_mc prior
        # for fgw calulation
        log10_mc_prior_scaled = (5/3*log10_mc_prior_scaled -
                                 log10_dL_scaled + np.log10(2*np.pi))
        if mask_range[1] - mask_range[0] > 0.1: 
            # log10_fgw prior - account for finite range
            log10_fgw_prior = np.array(mask_range)
            # scale for log10_h0 prior calculation
            log10_fgw_prior_scaled = 2/3*log10_fgw_prior
            # strain prior will be a convolution
            if upper_limit:
                log10_h0_prior_str = ('log10_h0:Convolve(LinearExp(pmin='
                                      f'{log10_mc_prior_scaled[0]}, '
                                      f'pmax={log10_mc_prior_scaled[1]}), '
                                      'Uniform(pmin='
                                      f'{log10_fgw_prior_scaled[0]}, '
                                      f'pmax={log10_fgw_prior_scaled[1]}))')
            else:
                log10_h0_prior_str = ('log10_h0:Convolve(Uniform(pmin='
                                      f'{log10_mc_prior_scaled[0]}, '
                                      f'pmax={log10_mc_prior_scaled[1]}), '
                                      'Uniform(pmin='
                                      f'{log10_fgw_prior_scaled[0]}, '
                                      f'pmax={log10_fgw_prior_scaled[1]}))')
        else:
            # calculate strain prior directly
            log10_h0_prior = log10_mc_prior_scaled + 2*np.mean(mask_range)/3
            if upper_limit:
                log10_h0_prior_str=('log10_h0:LinearExp(pmin='
                                    f'{log10_h0_prior[0]}'+
                                    f', pmax={log10_h0_prior[1]})')
            else:
                log10_h0_prior_str = (f'log10_h0:Uniform(pmin='
                                      f'{log10_h0_prior[0]}'+
                                      f', pmax={log10_h0_prior[1]})')
            
        prior_idx = [i for i in range(len(c_mask.priors)) if
                     'log10_h0' in c_mask.priors[i]][0]
        c_mask.priors[prior_idx] = log10_h0_prior_str
    
    mean_val = np.round(np.mean(mask_range),decimals=2)
    corner_single(target_str+f'_{masking_param}_{mean_val}', c_mask,
                  dataset, project_path, save_loc=save_loc, vary_crn=vary_crn,
                  vary_fgw=vary_fgw, detection=detection,
                  upper_limit=upper_limit, pub_fig=pub_fig,
                  target_fancy_name=target_fancy_name, truths=truths,
                  plot_mc_prior_mean=plot_mc_prior_mean, Npoints=Npoints)
    

def remove_old_chains(chaindir):
    dirs = glob.glob(f'{chaindir}/*')
    for d in dirs:
        # only remove old chains if I made a core
        if os.path.isfile(f'{d}/core.h5'):
            print(f'Removing subdirs for {chaindir}/{d.split("/")[-1]}')
            subdirs = glob.glob(f'{d}/*')
            [shutil.rmtree(sd, ignore_errors=True) for sd in subdirs]
        else:
            print(f'skipping {d} (no core.h5 found)')

    
def set_converged(c, source_name, path, detection=False,
                  overwrite=False, remove_oldchains=False):
    full_params = [c.params[i] for i in range(len(c.params[:-4])) if
                   'pmin' in c.priors[i]]
    emp_dist_file_name = f'{path}/empirical_dists/{source_name}_emp_dist.pkl'
    if not os.path.isfile(emp_dist_file_name) or overwrite:
        if overwrite and os.path.isfile(emp_dist_file_name):
            print('overwriting empirical distribution')
        make_empirical_distributions_from_core(c, full_params,
                                               filename=emp_dist_file_name)
    else:
        print('empirical distribution already exists!')
        
        
# Input core with FS red noise
def fs_bayesfac(c, gw_basename='crn', logAmin=-10):
    bf = np.zeros(30)
    bf_sig = np.zeros(30)
    for i in range(30):
        p = f'{gw_basename}_log10_rho_{i}'
        bf[i], bf_sig[i] = bayes_fac(c.get_param(p, to_burn=True),
                                     logAmin=logAmin, logAmax=-4)
    return bf, bf_sig
        
# free spectra
def free_spec(cs, psrs, gw_basename='crn', save_loc=None, colors=None,
              labels=None, violin=True, n_c=30, extra_save_label='',
              pmin=-10, title=None):
    Tspan = (np.max([np.max(p.toas) for p in psrs]) -
             np.min([np.min(p.toas) for p in psrs]))
    freqs = np.linspace(1/Tspan, n_c/Tspan, n_c)
    if not isinstance(cs, list):
        cs = [cs]
    for c in cs:
        c.set_rn_freqs(freqs, Tspan, n_c)

    fig, axes = plt.subplots(2,1,figsize=(8,5),sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.5]})

    ax, fig = plot_rednoise_spectrum('None', cs, gw_basename=gw_basename,
                                     Colors=colors,
                                     return_plot=True,
                                     free_spec_violin=violin,
                                     show_figure=False,
                                     labels=labels,
                                     plot_2d_hist=False, verbose=False,
                                     semilogx=False, fig=fig,
                                     set_title=False, axes=list(axes),
                                     legend=False, rasterized=True)
    
    axes[0].set_ylim([-9,-5])
    bfs = {}
    bf_sigs = {}
    masks = {}
    for c, label in zip(cs, labels):
        bfs[label], bf_sigs[label] = fs_bayesfac(c, gw_basename, logAmin=pmin)
        masks[label] = np.logical_or(np.isnan(bf_sigs[label]),
                                     np.isnan(bfs[label]))
    ylim = np.diff(axes[1].get_ylim())
    # BF lower limits
    max_ = 50
    # fix nans in BF estimates
    mfc = None
    for c, label, color in zip(cs, labels, colors):
        #bfs[ml][np.isnan(bfs[ml])] = max_
        for idx in range(len(masks[label])):
            if masks[label][idx]:
                bfs[label][idx] = np.nanmin([max_, bfs[label][idx]])

        # plot
        axes[1].errorbar(c.rn_freqs[~masks[label]],
                         np.log10(bfs[label][~masks[label]]),
                         yerr=bf_sigs[label][~masks[label]], fmt='o',
                         color=color, label=label, mfc=mfc, rasterized=True)
        axes[1].errorbar(c.rn_freqs[masks[label]],
                         np.log10(bfs[label][masks[label]]),
                         yerr=ylim[0]/10, lolims=True, fmt='o',
                         color=color, mfc=mfc, rasterized=True)
        mfc = 'none'
    axes[1].axvline(1/24/3600/365.25, color='0.3', ls='--',
                    rasterized=True)
    ylim = axes[1].get_ylim()
    textheight = ylim[0] + 0.5*(ylim[1] - ylim[0])
    axes[1].text(1/24/3600/365.25-1.7e-9, textheight, r'$f = 1$/yr',
                 rotation='vertical', fontsize='large', color='0.3')
    axes[1].grid(which='both', linestyle='--')
    axes[0].set_ylabel(r'$\log_{10}\rho_i$', fontsize='xx-large')
    axes[1].set_ylabel(r'$\log_{10}\mathcal{B}_i$', fontsize='xx-large')
    #axes[1].semilogx()
    axes[1].set_xticklabels(np.arange(-10,90,10))
    axes[1].set_xlabel('Frequency (nHz)', fontsize='x-large')
    if title:
        axes[0].set_title(title, fontsize=16)
    if len(cs) > 1:
        lines = [mlines.Line2D([], [], color=colors[i], label=labels[i])
                 for i in range(len(cs))]
        axes[0].legend(handles=lines, fontsize='large')
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    if save_loc:
        fig.savefig(f'{save_loc}/free_spec{extra_save_label}.pdf',
                    format='pdf', dpi=600, bbox_inches='tight')


