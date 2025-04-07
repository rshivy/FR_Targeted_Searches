import os
import pickle
import json

import numpy as np
from mpi4py import MPI

from enterprise.signals import gp_signals, white_signals, parameter, utils, selections, signal_base
from targeted_cws_ng15.Dists_Parameters import PXDistParameter
from QuickCW.PulsarDistPriors import DMDistParameter
from targeted_cws_ng15.new_delays_2 import cw_delay_new as cw_delay
from enterprise_extensions.deterministic import CWSignal

from enterprise_extensions.sampler import get_parameter_groups, get_cw_groups, group_from_params, save_runtime_info
from tsutils.utils import get_cw_groups
from PTMCMCSampler.PTMCMCSampler import PTSampler as Ptmcmc
from targeted_cws_ng15.jump_proposal import JumpProposal
import targeted_cws_ng15.Dists_Parameters as dists

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

psrpath = '/gpfs/gibbs/project/mingarelli/frh7/targeted_searches/data/ePSRs/ng15_v1p1/v1p1_de440_pint_bipm2019.pkl'
noisedict_path = 'noise_dicts/15yr_wn_dict.json'
psrdists_path = 'psr_distances/pulsar_distances_15yr.pkl'

outputdir = 'data/chains/ng15_v1p1/all_sky_ul'

if size > 1:
    chainpath = outputdir + '/chain_1.0.txt'
else:
    chainpath = outputdir + '/chain_1.txt'

n_samples = 0
resume = False
if os.path.exists(chainpath):
    with open(chainpath, 'r') as f:
        n_samples = len(f.readlines())
if n_samples >= 100:
    print(f'Continuing from n={n_samples} samples')
    resume = True

with open(psrpath, 'rb') as f:
    psrs = pickle.load(f)

tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
tspan = np.max(tmax) - np.min(tmin)
tref = max(tmax)

tm = gp_signals.MarginalizingTimingModel(use_svd=True)

cos_gwtheta = parameter.Uniform(pmin=-1, pmax=1)('cos_gwtheta')
gwphi = parameter.Uniform(pmin=0, pmax=2*np.pi)('gwphi')
log10_dist = parameter.Uniform(-2.0, 4.0)('log10_dL')
log10_fgw = parameter.Uniform(-9.0, -7.0)('log10_fgw')
log10_mc = parameter.LinearExp(7, 12)('log10_mc')
phase0 = parameter.Uniform(0, 2 * np.pi)('phase0')
psi = parameter.Uniform(0, np.pi)('psi')
cos_inc = parameter.Uniform(-1, 1)('cos_inc')

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

with open(psrdists_path, 'rb') as f:
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

x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)

# Set up initial covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

# Get parameter groups
crn_params = ['crn_log10_A', 'gamma_crn']
groups = get_parameter_groups(pta) + get_cw_groups(pta) + [group_from_params(pta, crn_params)]

# Intialize sampler
sampler = Ptmcmc(ndim=ndim,
                 logl=pta.get_lnlikelihood,
                 logp=pta.get_lnprior,
                 cov=cov,
                 groups=groups,
                 outDir=outputdir,
                 resume=resume)

# Create and add jump proposals
jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_red_prior, 30)
sampler.addProposalToCycle(jp.draw_from_cw_prior, 20)
sampler.addProposalToCycle(jp.draw_from_prior, 10)

sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_fgw']), 10)
sampler.addProposalToCycle(jp.draw_from_par_prior(['log10_mc']), 5)
sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 5)

jpCW = dists.JumpProposalCW(pta)
sampler.addProposalToCycle(jpCW.phase_psi_reverse_jump, 1)

pdist_pars = [p for p in pta.param_names if 'p_dist' in p]
pphase_pars = [p for p in pta.param_names if 'p_phase' in p]
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pdist_pars, 'p_dist'), 30)
sampler.addProposalToCycle(jpCW.draw_from_many_par_prior(pphase_pars, 'p_phase'), 30)
sampler.addAuxilaryJump(jpCW.fix_cyclic_pars)

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
    for signal_collection in pta.pulsarmodels:
        for signal in signal_collection.signals:
            for param_name, param in signal._params.items():  # The public attribute only returns nonconstant params
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
               DEweight=25,
               writeHotChains=True)
