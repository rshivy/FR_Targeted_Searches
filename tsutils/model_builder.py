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

from targeted_cws_ng15.Dists_Parameters import PXDistParameter
from QuickCW.PulsarDistPriors import DMDistParameter


def ts_model_builder(target_prior_path,
                     pulsar_path,
                     noisedict_path,
                     pulsar_dists_path,
                     exclude_pulsars=None,
                     vary_fgw=True):
    """
    Builds a PTA object according to my usual targeted search model choices

    :param target_prior_path: Path to a json file containing prior values for the target
    :param pulsar_path: Path to a pkl containing the NANOGrav pulsars
    :param noisedict_path: Path to a json file containing noise parameter valus
    :param pulsar_dists_path: Path to a pkl containing a dict of pulsar distance parameter values
    :param exclude_pulsars: A list of pulsar names to not use, default is None
    :param vary_fgw: False uses target prior value, True is log uniform +/- log10(6)
    """
    #####################
    # Set Target Priors #
    #####################

    with open(target_prior_path, 'rb') as f:
        target_priors = json.load(f)

    target_ra = target_priors['RA']
    target_dec = target_priors['DEC']
    target_log10_dist = target_priors['log10_dist']
    target_log10_freq = target_priors['log10_freq']

    target_coords = SkyCoord(target_ra, target_dec)
    target_coords.representation_type = 'physicsspherical'
    target_cos_theta = np.cos(target_coords.theta.to(u.rad))
    target_phi = target_coords.phi.to(u.rad)

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
    cos_gwtheta = parameter.Constant(val=target_cos_theta)('cos_gwtheta')  # position of source
    gwphi = parameter.Constant(val=target_phi)('gwphi')  # position of source
    log10_dist = parameter.Constant(val=target_log10_dist)('log10_dist')  # sistance to source
    if vary_fgw:  # Allow gw frequency to vary by a factor of six in either direction
        target_log10_freq_low = target_log10_freq - np.log10(6)
        target_log10_freq_high = target_log10_freq + np.log10(6)
        log10_fgw = parameter.Uniform(pmin=target_log10_freq_low,
                                      pmax=target_log10_freq_high)('log10_fgw')
    else:
        log10_fgw = parameter.Constant(val=target_log10_freq)('log10_fgw')

    log10_mc = parameter.Uniform(8, 11)('log10_mc')  # chirp mass of binary
    phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')  # gw phase
    psi = parameter.Uniform(0, np.pi)('psi')  # gw polarization
    cos_inc = parameter.Uniform(-1, 1)('cos_inc')  # inclination of binary with respect to Earth

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
    rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=tspan, selection=backend)

    # Common red noise
    log10_A_crn = parameter.Uniform(-18, -11)('crn_log10_A')
    gamma_crn = parameter.Uniform(0, 7)('gamma_crn')

    cpl = utils.powerlaw(log10_A=log10_A_crn, gamma=gamma_crn)

    crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=tspan, name='crn')

    s = tm + efeq + ec + rn + crn

    with open(pulsar_dists_path, 'rb') as f:
        psrdists = pickle.load(f)
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
                         scale_shift_pdists=False)  # Bjorn's toggle to fix pulsar distances

        cw = CWSignal(cw_wf, ecc=False, psrTerm=True)
        s += cw

    # Instantiate signal collection
    model = [s(psr) for psr in psrs]
    pta = signal_base.PTA(model)

    # Set white noise parameters
    with open(noisedict_path, 'r') as fp:
        noise_params = json.load(fp)
    pta.set_default_params(noise_params)

    return pta


if __name__ == '__main__':
    target_prior_path_def = 'Target_Priors/001_MCG_5-40-026.json'
    pulsar_path_def = ('/gpfs/gibbs/project/mingarelli/frh7/'
                       'targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl')
    noisedict_path_def = 'noise_dicts/15yr_wn_dict.json'
    pulsar_dists_path_def = 'psr_distances/pulsar_distances_15yr.pkl'
    exclude_pulsars_def = []

    pta = ts_model_builder(target_prior_path=target_prior_path_def,
                           pulsar_path=pulsar_path_def,
                           noisedict_path=noisedict_path_def,
                           pulsar_dists_path=pulsar_dists_path_def,
                           exclude_pulsars=exclude_pulsars_def,
                           vary_fgw=True)
