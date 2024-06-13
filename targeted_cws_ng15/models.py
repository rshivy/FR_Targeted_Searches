#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:11:46 2024

@author: blarsen10, adapted from Caitlin Witt
"""

import types
import numpy as np
import json, pickle
#import inspect
from astropy.coordinates import SkyCoord
from enterprise.signals import signal_base, gp_signals, parameter, utils
from enterprise.signals import gp_priors as gpp
from enterprise_extensions.blocks import (red_noise_block, white_noise_block)
#from enterprise_extensions.blocks import common_red_noise_block
from enterprise_extensions.deterministic import CWSignal
from enterprise_extensions import model_orfs
from QuickCW.PulsarDistPriors import DMDistParameter#, PXDistParameter
from targeted_cws_ng15.Dists_Parameters import PXDistParameter
from targeted_cws_ng15 import new_delays_2 as nd
#from targeted_cws_ng15 import Dists_Parameters as dists
# from DR3_noise_modeling.e_e_models import (dm_noise_block,
#                                            chromatic_noise_block,
#                                            solar_wind_block,
#                                            dm_exponential_dip)

def cw_model_2(psrs, priors, noisedict_path, psr_distance_path=None,
               orf=None, gw_psd='powerlaw', gw_components=14, gamma_val=None,
               log10_A_val=None, rn_components=30, white_vary=False,
               tnequad=False, tm_marg=True, tm_svd=True, tm_norm=True,
               is_wideband=False, rn_use_total_Tspan=True, gp_ecorr=False,
               log10_mc_prior='normal', normal_pdist_priors=False,
               vary_fgw=False, all_sky=False, fixedpoint=False, nogwb=False,
               free_p_phases=True):
    
    """
    psrs: list of pulsar objects
    priors: json of astrophysical CW priors
    noise_dict: path to dictionary of white noise parameters
        To avoid dependencies on datasets, can set to None and set params
        in a separate script
    orf: string
        supported values: 'hd' or None
    gw_psd: string
        supported values: ['powerlaw']
    gw_components: number of gw freqs
    gamma_val: None to vary gamma, or float to fixed value (i.e. 13/3)
    log10_A_val: None to vary log10_A, or float to fixed value
    rn_components: number of irn freqs
    white_vary: True to sample WN params
    tnequad: True to use tnequad convention, False for t2equad convention
    tm_marg: Marginalizing timing model (faster if holding WN fixed)
    tm_svd: SVD decomposition for timing model inversions
    tm_norm: Normalize timing model
    is_wideband: Wideband dataset
    rn_use_total_Tspan: set True to use total array Tspan for single psr red
        noise. False to use individual red noises
    log10_mc_priors: string
        supported values: ['normal', 'uniform']
    normal_pdist_priors: set True to just use normal psr distance priors using
        mean and sigma values in the distance file. Otherwise, set the priors
        to use inverse Gaussian for PX distances and box+half Gaussian for DM
        distances
    vary_fgw: set True to vary fGW as free param
    all_sky: set True to do an all-sky search
    fixedpoint: set True to vary only the achromatic red noise, holding the
        chromatic params fixed. The default is False. Does nothing if using
        standard noise models
    nogwb: Set True to turn OFF the common red noise model
    free_p_phases: Set True to sample an additional pulsar phase factor for
        all pulsars. False to calculate the pulsar term purely using pulsar
        distance
    """
    
    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    tref = max(tmax)
    if rn_use_total_Tspan:
        rn_Tspan = Tspan
    else:
        rn_Tspan = None
        
    # get noise params
    noise_params = {}
    with open(noisedict_path, 'r') as fp:
        noise_params.update(json.load(fp))
    if True:
        for param in list(noise_params.keys()):
            if 'ecorr' in param:
                par_new = param.replace(param.split('_')[0], param.split('_')[0]+'_basis_ecorr')
                noise_params[par_new] = noise_params[param]
        noise_params = {key: value for key, value in sorted(noise_params.items())}
    
    # Get distances
    with open(psr_distance_path, 'rb') as fp:
        dists = pickle.load(fp)
    
    # timing model
    if tm_marg:
        s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm)
        
    # white noise
    # by default this should use the fast sherman morrison method
    s += white_noise_block(vary=white_vary, inc_ecorr=True,
                           tnequad=tnequad, select='backend',
                           gp_ecorr=gp_ecorr)
    
    # common red noise
    if orf == 'hd':
        crn_name = 'gwb'
    elif not orf:
        crn_name = 'crn'
    if not nogwb:
        s += common_red_noise_block(psd=gw_psd, prior='log-uniform',
                                    Tspan=Tspan, components=gw_components,
                                    gamma_val=gamma_val, orf=orf,
                                    name=crn_name, log10_A_val=log10_A_val)
    
    # CW model
    # sky location (fixed)
    if all_sky:
        cos_gwtheta=parameter.Uniform(-1, 1)('cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)('gwphi')
    else:
        c = SkyCoord(priors['RA'], priors['DEC'], frame='icrs')
        cos_gwtheta=parameter.Constant(np.cos((np.pi)/2-c.dec.rad))('cos_gwtheta')
        gwphi = parameter.Constant(c.ra.rad)('gwphi')
    
    # frequency (fixed)
    if vary_fgw:
        log10_fgw = parameter.Uniform(-8, -7)('log10_fgw')
    else:
        log10_fgw = parameter.Constant(priors['log10_freq'])('log10_fgw')
    
    # luminosity distance (fixed)
    log10_dL = parameter.Constant(priors['log10_dist'])('log10_dL')
    tref = max(tmax)
    
    # chirp mass
    if log10_mc_prior == 'normal':
        log10_mc = parameter.Normal(mu=priors['log10_Mc'],
                                    sigma=priors['log10_Mc_sigma'])('log10_mc')
    elif log10_mc_prior == 'uniform':
        log10_mc = parameter.Uniform(8,11)('log10_mc')
    elif log10_mc_prior == 'uniform_low':
        log10_mc = parameter.Uniform(7,10)('log10_mc')
    elif log10_mc_prior == 'linearexp':
        log10_mc = parameter.LinearExp(7,11)('log10_mc')
    
    # phase/polarization/inclination params
    phase0 = parameter.Uniform(0, 2*np.pi)('phase0')
    psi = parameter.Uniform(0, np.pi)('psi')
    cos_inc = parameter.Uniform(-1, 1)('cos_inc')
    
    # pulsar term phase
    if free_p_phases:
        p_phase = parameter.Uniform(0, 2*np.pi)
    else:
        p_phase = None
    
    # Everything else depends on pulsar distance
    models = []
    for p in psrs:
        
        # custom noise
        s2 = s + custom_noise_block(p.name, noise_params,
                                    fixedpoint=fixedpoint, Tspan=rn_Tspan,
                                    rn_Nfreqs=rn_components)
        
        if normal_pdist_priors:
            p_dist = parameter.Normal(mu=dists[p.name][0],
                                      sigma=dists[p.name][1])
        elif dists[p.name][2] == 'PX':
            p_dist = PXDistParameter(dist=dists[p.name][0],
                                     err=dists[p.name][1])
        elif dists[p.name][2] == 'DM':
            p_dist = DMDistParameter(dist=dists[p.name][0],
                                     err=dists[p.name][1])
        else:
            raise ValueError("Can't find if distances are PX or DM!")
            
        # Waveform and CW signal for PX and DM distances
        wf = nd.cw_delay_new(cos_gwtheta=cos_gwtheta, gwphi=gwphi,
                             cos_inc=cos_inc, log10_mc=log10_mc,
                             log10_fgw=log10_fgw, log10_dist=log10_dL,
                             phase0=phase0, psi=psi, psrTerm=True,
                             p_dist=p_dist, p_phase=p_phase, evolve=True,
                             check=False, tref=tref)
        s2 += CWSignal(wf, ecc=False, psrTerm=True, name = 'cw')
        
        # act on psr objects
        models.append(s2(p))
    
    # set pta object
    pta = signal_base.PTA(models)
    
    # Read in and set white noise dictionary
    pta.set_default_params(noise_params)
    
    return pta


def gwb_only_model(psrs, noisedict_path, orf=None,
                   gw_psd='powerlaw', gw_components=14, gamma_val=None,
                   log10_A_val=None, rn_components=30, white_vary=False,
                   tnequad=False, tm_marg=True, tm_svd=True, tm_norm=True,
                   is_wideband=False, rn_use_total_Tspan=True,
                   fixedpoint=False):
    
    """
    psrs: list of pulsar objects
    orf: string
        supported values: 'hd' or None
    gw_psd: string
        supported values: ['powerlaw', 'turnover' 'spectrum']
    gw_components: number of gw freqs
    gamma_val: None to vary gamma, or float to fixed value (i.e. 13/3)
    log10_A_val: None to vary log10_A, or float to fixed value
    rn_components: number of irn freqs
    white_vary: True to sample WN params
    tnequad: True to use tnequad convention, False for t2equad convention
    tm_marg: Marginalizing timing model (faster if holding WN fixed)
    tm_svd: SVD decomposition for timing model inversions
    tm_norm: Normalize timing model
    is_wideband: Wideband dataset
    rn_use_total_Tspan: set True to use total array Tspan for single psr red
        noise. False to use individual red noises
    fixedpoint: set True to vary only the achromatic red noise, holding the
        chromatic params fixed. The default is False. Does nothing if using
        standard noise models
    """
    
    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    if rn_use_total_Tspan:
        rn_Tspan = Tspan
    else:
        rn_Tspan = None
        
    # get noise params
    noise_params = {}
    with open(noisedict_path, 'r') as fp:
        noise_params.update(json.load(fp))
    
    # timing model
    if tm_marg:
        s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm)
        
    # white noise
    # by default this should use the fast sherman morrison method
    s += white_noise_block(vary=white_vary, inc_ecorr=True,
                           tnequad=tnequad, select='backend')
    
    # common red noise
    if orf == 'hd':
        crn_name = 'gwb'
    elif not orf:
        crn_name = 'crn'
    s += common_red_noise_block(psd=gw_psd, prior='log-uniform', Tspan=Tspan,
                                components=gw_components, gamma_val=gamma_val,
                                orf=orf, name=crn_name,
                                log10_A_val=log10_A_val)
    
    models = []
    for p in psrs:
        # custom noise
        s2 = s + custom_noise_block(p.name, noise_params,
                                    fixedpoint=fixedpoint, Tspan=rn_Tspan,
                                    rn_Nfreqs=rn_components)
        
        # act on psr objects
        models.append(s2(p))
    
    # set pta object
    pta = signal_base.PTA(models)
    
    # Read in and set white noise dictionary
    pta.set_default_params(noise_params)
    
    return pta


# customize the red noise, and chromatic noise on a per-pulsar
# basis. You can also enter in the standard 15yr white noise dictionary and
# this will model the noise as normal (std red noise + white noise). Note the
# choice of using DMX or not comes separately by specifying a different pulsar
# pickle file, although we could also input the psrs to inform this...
def custom_noise_block(psrname, params, fixedpoint=False,
                       Tspan=None, rn_Nfreqs=30):
    '''

    Parameters
    ----------
    psrname : str
        Pulsar name. Use to parse params dictionary.
    params : dict
        Dictionary of max like params/settings for custom model.
    fixedpoint : bool, optional
        Select True to vary only the achromatic red noise, holding the
        chromatic params fixed. The default is False.
    rn_Tspan : float, optional
        Tspan to use for Fourier-basis GPs. The default is None, which means
        the individual pulsar timepsan will get used by default.
    rn_Nfreqs : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    None.

    '''
    
    # intrinsic red noise; priors are from (-20, -11) and (0, 7)
    if f'{psrname}_red_noise_Nfreqs' in params:
        rn_Nfreqs = params[f'{psrname}_red_noise_Nfreqs']
    noise = red_noise_block(psd='powerlaw', prior='log-uniform',
                            components=rn_Nfreqs, Tspan=Tspan,
                            gamma_val=None, select=None)
    
    # check if used DMGP, otherwise skip (assuming DMX in use)
    if f'{psrname}_dm_gp_Nfreqs' in params:
        dm_Nfreqs = params[f'{psrname}_dm_gp_Nfreqs']
        dm_log10_A = None
        dm_gamma = None
        if fixedpoint:
            dm_log10_A = params[f'{psrname}_dm_gp_log10_A']
            dm_gamma = params[f'{psrname}_dm_gp_gamma']
        # dm noise; Fourier-basis GP w/ power law
        # default priors are from (-20, -11) and (0, 7)
        noise += dm_noise_block(gp_kernel='diag', psd='powerlaw',
                                prior='log-uniform', components=dm_Nfreqs,
                                gamma_val=dm_gamma, log10_A_val=dm_log10_A,
                                Tspan=Tspan)

    # check if used chrom_gp, otherwise skip
    if f'{psrname}_chrom_gp_Nfreqs' in params:
        chrom_Nfreqs = params[f'{psrname}_chrom_gp_Nfreqs']
        chrom_log10_A = None
        chrom_gamma = None
        if fixedpoint:
            chrom_log10_A = params[f'{psrname}_chrom_gp_log10_A']
            chrom_gamma = params[f'{psrname}_chrom_gp_gamma']
        # dm noise; Fourier-basis GP w/ power law
        # default priors are from (-20, -11) and (0, 7)
        # idx = 4 by default
        noise += chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                                       prior='log-uniform', idx=4,
                                       components=chrom_Nfreqs, Tspan=Tspan,
                                       gamma_val=chrom_gamma,
                                       log10_A_val=chrom_log10_A)
    
    # check if used single pulsar solar wind
    if f'{psrname}_n_earth' in params:
        n_earth = None
        if fixedpoint:
            n_earth = params[f'{psrname}_n_earth']
        # deterministic model
        noise += solar_wind_block(n_earth=n_earth, ACE_prior=False,
                                  include_swgp=False, common_signal=False)

    # check if including exp dips
    # these are currently supported for J1713's 2 dip events, with input param
    # names "exp1_{param}" and "exp2_{param}"
    expdip_params = [par for par in params if f'{psrname}_exp' in par]
    if len(expdip_params) > 0:
        # fix chromatic indices by default. Trying to sample this during a
        # full PTA analysis is probably overkill and not worth the cost
        idxs = [params[f'{psrname}_exp{i}_idx'] for i in range(1,3)]
        basenames = ['exp1', 'exp2']
        # priors for varied run
        tmin = [54740, 57506]
        tmax = [54780, 57514]
        log10_tau_max = [3.5, 2.0]
        log10_A_min = [-6.5, -6.1]
        log10_A_max = [-5.4, -5.6]
        t, log10_tau, log10_A = [None,None], [None,None], [None,None]
        if fixedpoint:
            t = [params[f'{psrname}_exp{i}_t0'] for i in range(1,3)]
            log10_tau = [params[f'{psrname}_exp{i}_log10_tau'] for
                         i in range(1,3)]
            log10_A = [params[f'{psrname}_exp{i}_log10_Amp'] for
                       i in range(1,3)]
        for dd in range(2):
            # using custom chromatic exponential dip (idx = 2 for DM)
            noise += dm_exponential_dip(tmin=tmin[dd], tmax=tmax[dd],
                                        idx=idxs[dd], log10_tau_min=1.2,
                                        log10_tau_max=log10_tau_max[dd],
                                        log10_A_min=log10_A_min[dd],
                                        log10_A_max=log10_A_max[dd],
                                        sign='negative',
                                        name=basenames[dd],
                                        t=t[dd], log10_tau=log10_tau[dd],
                                        log10_A=log10_A[dd])
    
    return noise
    

# fix the constant log10_A definition
def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, components=30, combine=True,
                           log10_A_val=None, gamma_val=None, delta_val=None,
                           logmin=None, logmax=None,
                           orf=None, orf_ifreq=0, leg_lmax=5,
                           name='gw', coefficients=False,
                           pshift=False, pseed=None):
    """
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum', 'broken_powerlaw']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar.
    :param log10_A_val:
        Value of log10_A parameter for fixed amplitude analyses.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param delta_val:
        Value of spectral index for high frequencies in broken power-law
        and turnover models. By default spectral index is varied in range [0,7].\
    :param logmin:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param logmax:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function [default=5]
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process

    """

    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(-1.0, 1.0, size=7)('gw_orf_bin')),
            'zero_diag_hd': model_orfs.zero_diag_hd(),
            'zero_diag_bin_orf': model_orfs.zero_diag_bin_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_zero_diag')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre')),
            'zero_diag_legendre_orf': model_orfs.zero_diag_legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre_zero_diag'))}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover', 'turnover_knee', 'broken_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        elif logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            else:
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(-18, -14)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                    lfb=lfb_gw, lfk=lfk_gw,
                                    kappa=kappa_gw, delta=delta_gw)
    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)

        # checking if priors specified, otherwise give default values
        if logmin is None:
            logmin = -9
        if logmax is None:
            logmax = -4

        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(logmin, logmax,
                                               size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients, combine=combine,
                                        components=components, Tspan=Tspan,
                                        name=name, pshift=pshift, pseed=pseed)
    elif orf in orfs.keys():
        if orf == 'crn':
            crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients, combine=combine,
                                            components=components, Tspan=Tspan,
                                            name=name, pshift=pshift, pseed=pseed)
        else:
            crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf],
                                                  components=components, combine=combine,
                                                  Tspan=Tspan,
                                                  name=name, pshift=pshift,
                                                  pseed=pseed)
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.FourierBasisCommonGP(cpl, orf,
                                              components=components, combine=combine,
                                              Tspan=Tspan,
                                              name=name, pshift=pshift,
                                              pseed=pseed)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn

    