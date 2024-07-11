"""
Recreating the detection run of the 3C66B paper
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
from enterprise_extensions.deterministic import cw_delay, CWSignal
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

target_frequency = 60.4 * (10 ** -9)  # Herz
target_log10_freq = np.log10(target_frequency)

target_ra = '02h 23m 11.4112s'
target_dec = '+42d 59m 31.384s'
target_z = 0.02126
target_coords = SkyCoord(target_ra, target_dec, distance=target_z * cu.redshift)
target_dist = target_coords.distance.to(u.Mpc, cu.with_redshift(WMAP9, distance='luminosity')).value
target_log10_dist = np.log10(target_dist)
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
psrs = [psr for psr in psrs if psr.name != 'J1713+0747']

noisedict_path = 'noise_dicts/15yr_wn_dict.json'
psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

################
# Setup Output #
################

outputdir = 'data/chains/ng15_v1p1/3C66B_det'
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

cos_gwtheta = parameter.Constant(val=target_cos_theta)('cos_gwtheta')  # position of source
gwphi = parameter.Constant(val=target_phi)('gwphi')  # position of source
log10_fgw = parameter.Constant(val=target_log10_freq)('log10_fgw')  # gw frequency
log10_mc = parameter.Uniform(7, 10)('log10_mc')  # chirp mass of binary
phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')  # gw phase
psi = parameter.Uniform(0, np.pi)('psi')  # gw polarization
cos_inc = parameter.Uniform(-1, 1)('cos_inc')  # inclination of binary with respect to Earth

constant_params = [cos_gwtheta, gwphi, log10_fgw]
constant_param_names = [p.name for p in constant_params]
constant_param_values = [p.value if isinstance(p.value, np.float64) else p.value.value
                         for p in constant_params]
constant_params = {name: value for name, value in zip(constant_param_names, constant_param_values)}


cw_wf = cw_delay(cos_gwtheta=cos_gwtheta,
                 gwphi=gwphi,
                 log10_fgw=log10_fgw,
                 log10_mc=log10_mc,
                 phase0=phase0,
                 psi=psi,
                 cos_inc=cos_inc,
                 log10_dist=target_log10_dist,
                 tref=tref,
                 evolve=False,
                 psrTerm=True,
                 p_dist=0)

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

log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)

rn = gp_signals.FourierBasisGP(pl, components=30, Tspan=Tspan, selection=backend)

s = cw + efeq + ec + rn

model = [s(psr) for psr in psrs]
pta = signal_base.PTA(model)

with open(noisedict_path, 'r') as fp:
    noise_params = json.load(fp)
pta.set_default_params(noise_params)

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
sampler = Ptmcmc(ndim,
                 pta.get_lnlikelihood,
                 pta.get_lnprior,
                 cov,
                 groups=groups,
                 outDir=outputdir,
                 resume=False)

# Create and add jump proposals
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_red_prior, 6)
sampler.addProposalToCycle(jp.draw_from_cw_prior, 6)
sampler.addProposalToCycle(jp.draw_from_prior, 3)

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
    constant_parampath = outputdir + '/constant_params.json'
    with open(constant_parampath, 'w') as f:
        json.dump(constant_param_names, f, indent=4)

    constant_priorpath = outputdir + '/constant_priors.json'
    with open(constant_priorpath, 'w') as f:
        json.dump(constant_params, f, indent=4)

    save_runtime_info(pta,
                      outdir=outputdir,
                      human='Forrest H')

#########
# Begin #
#########

N = 1_000_000
sampler.sample(x0,
               N,
               SCAMweight=40,
               AMweight=20,
               DEweight=20)
