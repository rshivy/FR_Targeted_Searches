"""
Full targeted search script
"""
import numpy as np
import pickle
import os
import json

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9

from enterprise_extensions.sampler import get_parameter_groups, JumpProposal, save_runtime_info
from enterprise_extensions.deterministic import CWSignal  # , cw_delay
from targeted_cws_ng15.new_delays_2 import cw_delay_new as cw_delay
from targeted_cws_ng15.jump_proposal import JumpProposal
import enterprise.signals.parameter as parameter
from enterprise.signals import gp_signals, white_signals
from enterprise.signals import signal_base, utils
from enterprise.signals import selections

from PTMCMCSampler.PTMCMCSampler import PTSampler as Ptmcmc

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#################
# Target Priors #
#################

target_prior_path = 'Target_Priors/001_MCG_5-40-026.json'
with open(target_prior_path, 'rb') as f:
    target_priors = json.load(f)

target_ra = target_priors['RA']
target_dec = target_priors['DEC']
target_log10_dist = target_priors['log10_dist']
target_log10_freq = target_priors['log10_freq']
target_log10_freq_low = target_log10_freq - np.log10(6)
target_log10_freq_high = target_log10_freq + np.log10(6)
# target_z = 0.02126
target_coords = SkyCoord(target_ra, target_dec)
# target_dist = target_coords.distance.to(u.Mpc, cu.with_redshift(WMAP9, distance='luminosity')).value
# target_log10_dist = np.log10(target_dist)
target_coords.representation_type = 'physicsspherical'
target_cos_theta = np.cos(target_coords.theta.to(u.rad))
target_phi = target_coords.phi.to(u.rad)

################
# Data Sources #
################

# Load pulsars from pickle
psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
with open(psrpath, 'rb') as f:
    psrs = pickle.load(f)
# Exclude J1713+0747
# psrs = [psr for psr in psrs if psr.name != 'J1713+0747']

noisedict_path = 'noise_dicts/15yr_wn_dict.json'
psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

################
# Setup Output #
################

outputdir = 'data/chains/ng15_v1p1/001_MCG_5-40-026_det_varyfgw'
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

##########################
# Setup Enterprise Model #
##########################

tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)
tref = max(tmax)

tm = gp_signals.TimingModel()

# CW parameters
cos_gwtheta = parameter.Constant(val=target_cos_theta)('cos_gwtheta')  # position of source
gwphi = parameter.Constant(val=target_phi)('gwphi')  # position of source
log10_dist = parameter.Constant(val=target_log10_dist)('log10_dist')  # sistance to source
# log10_fgw = parameter.Constant(val=target_log10_freq)('log10_fgw')  # gw frequency
# Allow frequency to vary by a factor of six in either direction
log10_fgw = parameter.Uniform(pmin=target_log10_freq_low, pmax=target_log10_freq_high)('log10_fgw')
log10_mc = parameter.Uniform(7, 10)('log10_mc')  # chirp mass of binary
phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')  # gw phase
psi = parameter.Uniform(0, np.pi)('psi')  # gw polarization
cos_inc = parameter.Uniform(-1, 1)('cos_inc')  # inclination of binary with respect to Earth

# Distance parameter class
p_dist = parameter.Normal()
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
                 p_dist=p_dist)

cw = CWSignal(cw_wf, ecc=False, psrTerm=True)

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

# Red Noise
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan, selection=backend)

# Common red noise
log10_A_crn = parameter.Uniform(-18, -11)('crn_log10_A')
gamma_crn = parameter.Uniform(0, 7)('gamma_crn')

cpl = utils.powerlaw(log10_A=log10_A_crn, gamma=gamma_crn)

crn = gp_signals.FourierBasisGP(cpl, components=14, Tspan=Tspan, name='crn')

s = cw + efeq + ec + rn + crn

model = [s(psr) for psr in psrs]
pta = signal_base.PTA(model)

with open(noisedict_path, 'r') as fp:
    noise_params = json.load(fp)
pta.set_default_params(noise_params)

with open(psrdists_path, 'rb') as f:
    psrdists = pickle.load(f)
for signal_collection in pta._signalcollections:
    for signal in signal_collection._signals:
        for param_key, param in signal._params.items():
            if 'p_dist' in param_key:
                psrname = param_key.split('_')[0]
                signal._params[param_key] = parameter.Normal(psrdists[psrname][0], psrdists[psrname][1])(param.name)

#######################
# PTMCMCSampler Setup #
#######################


# Get initial sample
x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

# Set up initial covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# Get parameter groups
groups = get_parameter_groups(pta)

# Intialize sampler
sampler = Ptmcmc(ndim=ndim,
                 logl=pta.get_lnlikelihood,
                 logp=pta.get_lnprior,
                 cov=cov,
                 groups=groups,
                 outDir=outputdir,
                 resume=False)

# Create and add jump proposals
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_red_prior, 30)
sampler.addProposalToCycle(jp.draw_from_cw_prior, 20)
sampler.addProposalToCycle(jp.draw_from_prior, 10)

sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_fgw']), 10)
sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_mc']), 5)
sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 5)

###################################
# Save everything before starting #
###################################

if rank == 0:
    # Parsing out the details of the parameters. This might be overly complicated but it works for now
    params = pta.params
    param_names = []
    param_details = {}
    for param in params:
        param_name, _, type_with_args = str(param).partition(':')
        param_type, _, args = type_with_args[:-1].partition('(')
        arg1, _, arg2 = args.partition(', ')
        param_names += [param_name]
        param_details[param_name] = [param_type, arg1, arg2]

    model_parampath = outputdir + '/model_params.json'
    with open(model_parampath, 'w') as f:
        json.dump(param_names, f, indent=4)

    model_priorpath = outputdir + '/model_priors.json'
    with open(model_priorpath, 'w') as f:
        json.dump(param_details, f, indent=4)

    # Also save the constant parameters

    constant_params = set()
    for signal_collection in pta._signalcollections:
        for signal in signal_collection._signals:
            for param_name, param in signal._params.items():
                if isinstance(param, parameter.ConstantParameter):
                    constant_params.add(param)
    constant_params = {cp.name: cp.value if isinstance(cp.value, (np.float64, float))
    else cp.value.value
                       for cp in sorted(list(constant_params), key=lambda cp: cp.name)}

    constant_parampath = outputdir + '/constant_params.json'
    with open(constant_parampath, 'w') as f:
        json.dump(list(constant_params.keys()), f, indent=4)

    constant_priorpath = outputdir + '/constant_priors.json'
    with open(constant_priorpath, 'w') as f:
        json.dump(constant_params, f, indent=4)

    save_runtime_info(pta=pta,
                      outdir=outputdir,
                      human='Forrest H')

#########
# Begin #
#########

N = 10_000_000
sampler.sample(p0=x0,
               Niter=N,
               SCAMweight=40,
               AMweight=25,
               DEweight=25)
