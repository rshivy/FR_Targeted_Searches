import json
import pickle

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from enterprise.signals import gp_signals, white_signals
from enterprise.signals import parameter
from enterprise_extensions.deterministic import CWSignal  # , cw_delay
from targeted_cws_ng15.new_delays_2 import cw_delay_new as cw_delay
from enterprise.signals import selections
from enterprise.signals import signal_base, utils
from enterprise.signals.gp_priors import broken_powerlaw

from targeted_cws_ng15.Dists_Parameters import PXDistParameter
from QuickCW.PulsarDistPriors import DMDistParameter

import matplotlib.pyplot as plt


def set_cw_params(target_prior_path, mass_prior, freq_prior, tspan):
    target_priors = load_target_priors(target_prior_path)

    cos_gwtheta = parameter.Constant(val=target_priors['cos_theta'])('cos_gwtheta')  # position of source
    gwphi = parameter.Constant(val=target_priors['phi'])('gwphi')  # position of source
    log10_dist = parameter.Constant(val=target_priors['log10_dist'])('log10_dist')  # distance to source
    if freq_prior == 'narrow':  # Allow gw frequency to vary by a factor of six in either direction
        target_log10_freq_low = target_priors['log10_freq'] - np.log10(2)
        target_log10_freq_high = target_priors['log10_freq'] + np.log10(3)
        log10_fgw = parameter.Uniform(pmin=target_log10_freq_low,
                                      pmax=target_log10_freq_high)('log10_fgw')
    elif freq_prior == 'full':
        log10_fgw = parameter.Uniform(pmin=np.log10(1 / tspan), pmax=-7)('log10_fgw')
    elif freq_prior == 'constant':
        log10_fgw = parameter.Constant(val=target_priors['log10_freq'])('log10_fgw')
    else:
        raise ValueError(f'Unknown value for vary_fgw: {freq_prior}.'
                         'options are {\'constant\', \'narrow\', \'full\'}')
    if mass_prior == 'detection':
        log10_mc = parameter.Uniform(7, 11)('log10_mc')  # chirp mass of binary
    elif mass_prior == 'upper-limit':
        log10_mc = parameter.LinearExp(7, 12)('log10_mc')
    else:
        raise ValueError(f'Unknown value for mass_prior: {mass_prior}.'
                         'options are {\'detection\', \'upper-limit\'}')
    phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')  # gw phase
    psi = parameter.Uniform(0, np.pi)('psi')  # gw polarization
    cos_inc = parameter.Uniform(-1, 1)('cos_inc')  # inclination of binary with respect to Earth

    cw_params = {'cos_gwtheta': cos_gwtheta,
                 'gwphi': gwphi,
                 'log10_dist': log10_dist,
                 'log10_fgw': log10_fgw,
                 'log10_mc': log10_mc,
                 'phase0': phase0,
                 'psi': psi,
                 'cos_inc': cos_inc, }

    return cw_params


def load_target_priors(target_prior_path):
    with open(target_prior_path, 'rb') as f:
        target_priors = json.load(f)

    target_ra = target_priors['RA']
    target_dec = target_priors['DEC']
    target_log10_dist = target_priors['log10_dist']
    target_log10_freq = target_priors['log10_freq']

    target_coords = SkyCoord(target_ra, target_dec)
    target_coords.representation_type = 'physicsspherical'
    target_cos_theta = np.cos(target_coords.theta.to(u.rad))
    target_phi = target_coords.phi.to(u.rad).value

    target_priors = {'cos_theta': target_cos_theta,
                     'phi': target_phi,
                     'log10_dist': target_log10_dist,
                     'log10_freq': target_log10_freq}
    return target_priors


def ts_broken_powerlaw(target_prior_path,
                       pulsar_path,
                       noisedict_path,
                       pulsar_dists_path,
                       exclude_pulsars=None,
                       vary_fgw='narrow',
                       mass_prior='detection'):

    with open(pulsar_path, 'rb') as f:
        psrs = pickle.load(f)
    # Exclude specified pulsars, if any
    if exclude_pulsars is not None:
        psrs = [psr for psr in psrs if psr.name not in exclude_pulsars]

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    tspan = np.max(tmax) - np.min(tmin)
    tref = max(tmax)

    tm = gp_signals.MarginalizingTimingModel(use_svd=True)

    cw_params = set_cw_params(target_prior_path, mass_prior, vary_fgw, tspan)
    cos_gwtheta = cw_params['cos_gwtheta']
    gwphi = cw_params['gwphi']
    log10_dist = cw_params['log10_dist']
    log10_fgw = cw_params['log10_fgw']
    log10_mc = cw_params['log10_mc']
    phase0 = cw_params['phase0']
    psi = cw_params['psi']
    cos_inc = cw_params['cos_inc']

    # White noise
    backend = selections.Selection(selections.by_backend)
    backend_ng = selections.Selection(selections.nanograv_backends)

    efac = parameter.Constant()
    log10_equad = parameter.Constant()
    log10_ecorr = parameter.Constant()
    efeq = white_signals.MeasurementNoise(efac=efac,
                                          log10_t2equad=log10_equad,
                                          selection=backend)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr,
                                        selection=backend_ng)

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=tspan)

    # Common red noise
    log10_A_crn = parameter.Uniform(-18, -11)('log10_A_crn')
    gamma_crn = parameter.Uniform(0, 7)('gamma_crn')
    delta_crn = parameter.Uniform(0, 7)('delta_crn')
    kappa_crn = parameter.Uniform(0.01, 0.5)('kappa_crn')
    log10_fb_crn = parameter.Uniform(-10, -7)('log10_fb_crn')

    cpl = broken_powerlaw(log10_A=log10_A_crn, gamma=gamma_crn, delta=delta_crn, log10_fb=log10_fb_crn, kappa=kappa_crn)

    crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=tspan, name='crn')

    s = tm + efeq + ec + rn + crn

    with open(pulsar_dists_path, 'rb') as f:
        psrdists = pickle.load(f)
    p_phase = parameter.Uniform(0, 2 * np.pi)
    signal_collections = []
    # Adding individual cws so we can set pulsar distances
    for psr in psrs:
        # Set pulsar distance parameter
        if psrdists[psr.name][2] == 'PX':
            p_dist = PXDistParameter(dist=psrdists[psr.name][0],
                                     err=psrdists[psr.name][1])
        elif psrdists[psr.name][2] == 'DM':
            p_dist = DMDistParameter(dist=psrdists[psr.name][0],
                                     err=psrdists[psr.name][1])
        else:
            raise ValueError("Can't find if distances are PX or DM!")

        cw_wf = cw_delay(cos_gwtheta=cos_gwtheta,
                         gwphi=gwphi,
                         log10_fgw=log10_fgw,
                         log10_mc=log10_mc,
                         phase0=phase0,
                         psi=psi,
                         cos_inc=cos_inc,
                         log10_dist=log10_dist,
                         tref=tref,
                         evolve=True,
                         psrTerm=True,
                         p_dist=p_dist,
                         p_phase=p_phase,
                         scale_shift_pdists=False)  # Bjorn's toggle to fix pulsar distances

        cw = CWSignal(cw_wf, ecc=False, psrTerm=True)
        signal_collection = s + cw
        signal_collections += [signal_collection(psr)]

    # Instantiate signal collection
    pta = signal_base.PTA(signal_collections)

    # Set white noise parameters
    with open(noisedict_path, 'r') as fp:
        noise_params = json.load(fp)
    pta.set_default_params(noise_params)

    return pta


def ts_model_builder(target_prior_path,
                     pulsar_path,
                     noisedict_path,
                     pulsar_dists_path,
                     exclude_pulsars=None,
                     vary_fgw='narrow',
                     mass_prior='detection'):
    """
    Builds a PTA object according to my usual targeted search model choices

    :param target_prior_path: Path to a json file containing prior values for the target
    :param pulsar_path: Path to a pkl containing the NANOGrav pulsars
    :param noisedict_path: Path to a json file containing noise parameter valus
    :param pulsar_dists_path: Path to a pkl containing a dict of pulsar distance parameter values
    :param exclude_pulsars: A list of pulsar names to not use, default is None
    :param vary_fgw: Options are {'constant', 'narrow', and 'full'} narrow is log uniform (1,6)*EM freq
    :param mass_prior: Options are {'detection', 'upper_limit'} corresponding to log uniform and uniform respectively
    """

    ################
    # Load Pulsars #
    ################

    with open(pulsar_path, 'rb') as f:
        psrs = pickle.load(f)
    # Exclude specified pulsars, if any
    if exclude_pulsars is not None:
        psrs = [psr for psr in psrs if psr.name not in exclude_pulsars]

    ##########################
    # Setup Enterprise Model #
    ##########################

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    tspan = np.max(tmax) - np.min(tmin)
    tref = max(tmax)

    tm = gp_signals.MarginalizingTimingModel(use_svd=True)

    # CW parameters
    cw_params = set_cw_params(target_prior_path, mass_prior, vary_fgw, tspan)
    cos_gwtheta = cw_params['cos_gwtheta']
    gwphi = cw_params['gwphi']
    log10_dist = cw_params['log10_dist']
    log10_fgw = cw_params['log10_fgw']
    log10_mc = cw_params['log10_mc']
    phase0 = cw_params['phase0']
    psi = cw_params['psi']
    cos_inc = cw_params['cos_inc']

    # White noise
    backend = selections.Selection(selections.by_backend)
    backend_ng = selections.Selection(selections.nanograv_backends)

    efac = parameter.Constant()
    log10_equad = parameter.Constant()
    log10_ecorr = parameter.Constant()
    efeq = white_signals.MeasurementNoise(efac=efac,
                                          log10_t2equad=log10_equad,
                                          selection=backend)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr,
                                        selection=backend_ng)

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=tspan)

    # Common red noise
    log10_A_crn = parameter.Uniform(-18, -11)('crn_log10_A')
    gamma_crn = parameter.Uniform(0, 7)('gamma_crn')

    cpl = utils.powerlaw(log10_A=log10_A_crn, gamma=gamma_crn)

    crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=tspan, name='crn')

    s = tm + efeq + ec + rn + crn

    with open(pulsar_dists_path, 'rb') as f:
        psrdists = pickle.load(f)
    p_phase = parameter.Uniform(0, 2 * np.pi)
    signal_collections = []
    # Adding individual cws so we can set pulsar distances
    for psr in psrs:
        # Set pulsar distance parameter
        if psrdists[psr.name][2] == 'PX':
            p_dist = PXDistParameter(dist=psrdists[psr.name][0],
                                     err=psrdists[psr.name][1])
        elif psrdists[psr.name][2] == 'DM':
            p_dist = DMDistParameter(dist=psrdists[psr.name][0],
                                     err=psrdists[psr.name][1])
        else:
            raise ValueError("Can't find if distances are PX or DM!")

        cw_wf = cw_delay(cos_gwtheta=cos_gwtheta,
                         gwphi=gwphi,
                         log10_fgw=log10_fgw,
                         log10_mc=log10_mc,
                         phase0=phase0,
                         psi=psi,
                         cos_inc=cos_inc,
                         log10_dist=log10_dist,
                         tref=tref,
                         evolve=True,
                         psrTerm=True,
                         p_dist=p_dist,
                         p_phase=p_phase,
                         scale_shift_pdists=False)  # Bjorn's toggle to fix pulsar distances

        cw = CWSignal(cw_wf, ecc=False, psrTerm=True)
        signal_collection = s + cw
        signal_collections += [signal_collection(psr)]

    # Instantiate signal collection
    pta = signal_base.PTA(signal_collections)

    # Set white noise parameters
    with open(noisedict_path, 'r') as fp:
        noise_params = json.load(fp)
    pta.set_default_params(noise_params)

    return pta

def mock_ts_model_builder(target_prior_path,
                     pulsar_path,
                     noisedict_path,
                     pulsar_dists_path,
                     exclude_pulsars=None,
                     vary_fgw='narrow',
                     mass_prior='detection'):
    """
    Builds a PTA object according to my usual targeted search model choices

    :param target_prior_path: Path to a json file containing prior values for the target
    :param pulsar_path: Path to a pkl containing the NANOGrav pulsars
    :param noisedict_path: Path to a json file containing noise parameter valus
    :param pulsar_dists_path: Path to a pkl containing a dict of pulsar distance parameter values
    :param exclude_pulsars: A list of pulsar names to not use, default is None
    :param vary_fgw: Options are {'constant', 'narrow', and 'full'} narrow is log uniform (1,6)*EM freq
    :param mass_prior: Options are {'detection', 'upper_limit'} corresponding to log uniform and uniform respectively
    """

    ################
    # Load Pulsars #
    ################

    with open(pulsar_path, 'rb') as f:
        psrs = pickle.load(f)
    # Exclude specified pulsars, if any
    if exclude_pulsars is not None:
        psrs = [psr for psr in psrs if psr.name not in exclude_pulsars]

    ##########################
    # Setup Enterprise Model #
    ##########################

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    tspan = np.max(tmax) - np.min(tmin)
    tref = max(tmax)

    tm = gp_signals.MarginalizingTimingModel(use_svd=True)

    # CW parameters
    cw_params = set_cw_params(target_prior_path, mass_prior, vary_fgw, tspan)
    cos_gwtheta = cw_params['cos_gwtheta']
    gwphi = cw_params['gwphi']
    log10_dist = cw_params['log10_dist']
    log10_fgw = cw_params['log10_fgw']
    log10_mc = cw_params['log10_mc']
    phase0 = cw_params['phase0']
    psi = cw_params['psi']
    cos_inc = cw_params['cos_inc']

    # White noise
    backend = selections.Selection(selections.by_backend)
    backend_ng = selections.Selection(selections.nanograv_backends)

    efac = parameter.Constant()
    log10_equad = parameter.Constant()
    log10_ecorr = parameter.Constant()
    efeq = white_signals.MeasurementNoise(efac=efac,
                                          log10_t2equad=log10_equad)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr)

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)

    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=tspan)

    # Common red noise
    log10_A_crn = parameter.Uniform(-18, -11)('crn_log10_A')
    gamma_crn = parameter.Uniform(0, 7)('gamma_crn')

    cpl = utils.powerlaw(log10_A=log10_A_crn, gamma=gamma_crn)

    crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=tspan, name='crn')

    s = tm + efeq + ec + rn + crn

    p_phase = parameter.Uniform(0, 2 * np.pi)
    p_dist = parameter.Normal(0, 1)
    signal_collections = []
    # Adding individual cws so we can set pulsar distances
    for psr in psrs:
        cw_wf = cw_delay(cos_gwtheta=cos_gwtheta,
                         gwphi=gwphi,
                         log10_fgw=log10_fgw,
                         log10_mc=log10_mc,
                         phase0=phase0,
                         psi=psi,
                         cos_inc=cos_inc,
                         log10_dist=log10_dist,
                         tref=tref,
                         evolve=True,
                         psrTerm=True,
                         p_phase=p_phase,
                         p_dist=p_dist)

        cw = CWSignal(cw_wf, ecc=False, psrTerm=True)
        signal_collection = s + cw
        signal_collections += [signal_collection(psr)]

    # Instantiate signal collection
    pta = signal_base.PTA(signal_collections)

    # Set white noise parameters
    with open(noisedict_path, 'r') as fp:
        noise_params = json.load(fp)
    pta.set_default_params(noise_params)

    return pta
